import random
import torch
import math
import sys
from timeit import default_timer
from neuralop import Trainer, LpLoss
import neuralop.mpu.comm as comm
from torch.cuda import amp
import wandb
import torch.nn.functional as F
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.datasets.tensor_dataset import TensorDataset

def bilinear_interpolate(x, res_tuple):
    return F.interpolate(x, size=res_tuple, mode='bilinear')

# Adapted from https://github.com/bogdanraonic3/ConvolutionalNeuralOperator/blob/main/Error_Distribution_VaryingResolution.py
def resize(x, out_size, channel_dim_last=False):
    if channel_dim_last:
        x = x.permute(0, 3, 1, 2)
        
    f = torch.fft.rfft2(x, norm='backward')
    f_z = torch.zeros((*x.shape[:-2], out_size[0], out_size[1]//2 + 1), dtype=f.dtype, device=f.device)
    # 2k+1 -> (2k+1 + 1) // 2 = k+1 and (2k+1)//2 = k
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    # 2k -> (2k + 1) // 2 = k and (2k)//2 = k
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    # x_z = torch.fft.ifft2(f_z, s=out_size).real
    x_z = torch.fft.irfft2(f_z, s=out_size).real
    x_z = x_z * (out_size[0] / x.shape[-2]) * (out_size[1] / x.shape[-1])
 
    # f_z[..., -f.shape[-2]//2:, :f.shape[-1]] = f[..., :f.shape[-2]//2+1, :]
 
    if channel_dim_last:
        x_z = x_z.permute(0, 2, 3, 1)
    return x_z

class LpLossAbs(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def square_abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return torch.square(diff)

    def rel(self, x, y):

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.square_abs(y_pred, y)

# https://stackoverflow.com/questions/70849287/how-to-merge-multiple-iterators-to-get-a-new-iterator-which-will-iterate-over-th
def combine_loaders_random(loaders):
    loaders = [load._get_iterator() for load in loaders]
    while loaders:
        it = random.choice(loaders)
        try:
            #yield next(enumerate(it))[1]
            yield next(it)
        except StopIteration:
            loaders.remove(it)

class MultiResTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(
        self,
        train_loaders,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_every: int=None,
        save_best: int=None,
        save_dir="./ckpt",
        resume_from_dir=None,
    ):
        """
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)
        
        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction'):
            if training_loss.reduction == "mean":
                warnings.warn(f"{training_loss.reduction=}. This means that the loss is "
                              "initialized to average across the batch dim. The Trainer "
                              "expects losses to sum across the batch dim.")

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert self.save_best in metrics,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f'Training on {[len(loader.dataset) for loader in train_loaders.values()]} samples'
                  f'         on resolutions {[name for name in train_loaders]}.')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        
        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loaders, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time
            )
            
            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                eval_losses=eval_losses,
                                                test_loaders=test_loaders)

                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loaders, training_loss):
        """
            train_loaders is now a dict mapping name : loader
        """
        train_loaders_copy = train_loaders.copy()
        combined_train_loaders = combine_loaders_random(list(train_loaders_copy.values()))
        n_train = sum([len(loader.dataset) for loader in train_loaders.values()])

        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        
        # track number of training examples in batch
        self.n_samples = 0

        for idx, sample in enumerate(combined_train_loaders):
            
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= n_train
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

# Adapted from Neural Operator library
def load_navier_stokes_pt(data_path, train_resolution,
                          n_train, n_tests,
                          batch_size, test_batch_sizes,
                          test_resolutions,
                          encode_input=True,
                          encode_output=True,
                          encoding='channel-wise',
                          channel_dim=1,
                          num_workers=2,
                          pin_memory=True, 
                          persistent_workers=True                          ):
    """Load the Navier-Stokes dataset
    """
    #assert train_resolution == 128, 'Loading from pt only supported for train_resolution of 128'

    train_resolution_str = str(train_resolution)

    data = torch.load(Path(data_path).joinpath('nsforcing_' + train_resolution_str + '_train.pt').as_posix())
    x_train = data['x'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    y_train = data['y'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    del data

    idx = test_resolutions.index(train_resolution)
    test_resolutions.pop(idx)
    n_test = n_tests.pop(idx)
    test_batch_size = test_batch_sizes.pop(idx)

    data = torch.load(Path(data_path).joinpath('nsforcing_' + train_resolution_str + '_test.pt').as_posix())
    x_test = data['x'][:n_test, :, :].unsqueeze(channel_dim).clone()
    y_test = data['y'][:n_test, :, :].unsqueeze(channel_dim).clone()
    del data
    
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        x_train = input_encoder(x_train)
        x_test = input_encoder(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        y_train = output_encoder(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    test_loaders = {train_resolution: test_loader}

    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        x_test, y_test = _load_navier_stokes_test_HR(data_path, n_test, resolution=res, channel_dim=channel_dim)
        if input_encoder is not None:
            x_test = input_encoder(x_test)

        test_db = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                  batch_size=test_batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        test_loaders[res] = test_loader

    return train_loader, test_loaders, output_encoder


def _load_navier_stokes_test_HR(data_path, n_test, resolution=256,
                                channel_dim=1,
                               ):
    """Load the Navier-Stokes dataset
    """
    if resolution == 64:
        downsample_factor = 16
    elif resolution == 128:
        downsample_factor = 8
    elif resolution == 256:
        downsample_factor = 4
    elif resolution == 512:
        downsample_factor = 2
    elif resolution == 1024:
        downsample_factor = 1
    else:
        raise ValueError(f'Invalid resolution, got {resolution}, expected one of [64, 128, 256, 512, 1024].')
    
    data = torch.load(Path(data_path).joinpath('nsforcing_1024_test1.pt').as_posix())

    if not isinstance(n_test, int):
        n_samples = data['x'].shape[0]
        n_test = int(n_samples*n_test)
        
    x_test = data['x'][:n_test, ::downsample_factor, ::downsample_factor].unsqueeze(channel_dim).clone()
    y_test = data['y'][:n_test, ::downsample_factor, ::downsample_factor].unsqueeze(channel_dim).clone()
    del data

    return x_test, y_test