################## ##################
# Based on https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_navier_stokes.py 
################## ##################
import torch
from torch import nn
import torch.nn.functional as F
import os

import argparse
from configmypy import ConfigPipeline, YamlConfig
from neuralop import Trainer, LpLoss
from utilities import load_navier_stokes_pt
from neuralop.utils import count_model_params
from utilities import resize

class InputOutputInterpolate(nn.Module):
    """
    Wrapper class that converts an NN (e.g., U-Net) into a neural operator in the naive way:
    by interpolating the input/output as necessary. 

    Given the training resolution, if (1) the input is lower resolution, up-sample
    the input, apply the pre-trained NN, then downsample. If (2) the input is
    higher resolution, down-sample the input, apply the pre-trained NN, then
    upsample the model output.
    """

    def __init__(self, model, train_res, interpolation='bilinear'):
        super().__init__()
        self.model = model
        self.train_res = int(train_res)

        if interpolation == 'bilinear':
            self.interpolate = lambda x, res_tuple: F.interpolate(x, size=res_tuple, mode=interpolation)
        elif interpolation == 'fourier':
            self.interpolate = lambda x, res_tuple: resize(x, res_tuple)
        else:
            raise NotImplementedError("Interpolation method not implemented.")

    def forward(self, x, **kwargs):
        # Assumes x is of shape (batch, channel, nx, ny)
        assert x.shape[-1] == x.shape[-2]
        res = int(x.shape[-1])
        
        if res == self.train_res:
            return self.model(x)
        else:
            assert self.train_res % res == 0 or res % self.train_res == 0
            new_input = self.interpolate(x, (self.train_res, self.train_res))
            train_res_output = model(new_input)
            output = self.interpolate(train_res_output, (res, res))
            return output            

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Model setup arguments.")
    parser.add_argument(
        'weights_path',
        type=str,
        help='Name of the model weights file in ./ckpts'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Name of the config file in ./configs'
    )
    parser.add_argument(
        '--interpolation',
        choices=['fourier', 'bilinear'],
        default='bilinear',
        help="Interpolation method to use (default: bilinear)"
    )
    args = parser.parse_args()

    # Read the configuration
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./" + args.config_path, config_name="ns", config_folder="./configs"
            )
        ]
    )
    config = pipe.read_conf()
    config_name = pipe.steps[-1].config_name

    model_name = config["arch"]

    # Loading the Navier-Stokes dataset
    train_loader, test_loaders, output_encoder = load_navier_stokes_pt(
        config.data.folder,
        train_resolution=config.data.train_resolution,
        n_train=config.data.n_train,
        batch_size=config.data.batch_size,
        test_resolutions=config.data.test_resolutions,
        n_tests=config.data.n_tests,
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
    )

    # Set up model
    pre_trained_model = torch.load(os.path.join('./ckpts', args.weights_path), weights_only=False)
    model = InputOutputInterpolate(pre_trained_model, config.data.train_resolution, args.interpolation)
    print("Model:", model_name)
    print("Model params:", count_model_params(model))
    print()

    eval_losses = {"l2": LpLoss(d=2, p=2)}

    trainer = Trainer(
        model=model,
        data_processor=None,
        n_epochs=config.opt.n_epochs,
        device=device,
        wandb_log=config.wandb.log,
        eval_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose
    )

    msg = "Losses"
    print("Model:", model_name)
    print("Weight path:", args.weights_path)
    print("Config file:", args.config_path)
    print("Interpolation:", args.interpolation)
    for name, loader in test_loaders.items():
        errors = trainer.evaluate(eval_losses, loader, log_prefix=name)
        for loss_name, loss_value in errors.items():
            msg += f' | {loss_name}={loss_value:.4f}'
    print(msg)