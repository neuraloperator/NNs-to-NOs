################## ##################
# Based on https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_navier_stokes.py 
################## ##################
import sys, os
import torch

from configmypy import ConfigPipeline, YamlConfig
from argparse import ArgumentParser
from neuralop import Trainer, get_model, LpLoss
from neuralop.utils import count_model_params

from models.unet import Unet
from models.vit import ViT
from transformers import ViTConfig
from models.unet_kernel_interp import UNetWithScale, UNet_kernel_interpolate
from models.oformer import OFormer, OFormerWrapper

from utilities import LpLossAbs, load_navier_stokes_pt

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find config name
    parser = ArgumentParser(description="Training script")
    parser.add_argument("config_path", type=str, help="Path of the YAML config file inside ./configs")
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
    save_name = model_name + '_ckpt_single_res'

    # Create output directory
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')

    # Print config to screen
    if config.verbose:
        pipe.log()
        sys.stdout.flush()

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
    if model_name.lower() == 'unet':
        hyperparams = config.unet
        model = Unet(
                    n_input_scalar_components = 1,
                    n_input_vector_components = 0,
                    n_output_scalar_components = 1,
                    n_output_vector_components = 0,
                    time_history = 1,
                    time_future = 1,
                    kernel_size = hyperparams.kernel_size,
                    hidden_channels = hyperparams.hidden_channels,
                    activation = 'gelu',
                    norm = True,                                    
                    ch_mults = hyperparams.ch_mults,
                    is_attn = hyperparams.is_attn,
                    mid_attn = False,
                    n_blocks = 2,
                    use1x1 = False,
                    num_groups = hyperparams.num_groups,
                    positional_embedding=hyperparams.positional_embedding
                ).to(device)
    elif model_name.lower() == 'vit':
        hyperparams = config.vit

        model_config = ViTConfig(
            image_size = config.data.train_resolution,
            num_channels = hyperparams.num_channels,
            patch_size = hyperparams.patch_size,
            encoder_stride = hyperparams.encoder_stride,
            hidden_size = hyperparams.hidden_size,
            num_hidden_layers = hyperparams.num_hidden_layers,
            num_attention_heads = hyperparams.num_attention_heads,
            intermediate_size = hyperparams.intermediate_size,
            hidden_act = hyperparams.hidden_act
        )

        model = ViT(model_config).to(device)
    elif model_name.lower() == 'unet_kernel_interp':
        hyperparams = config.unet_kernel_interp

        model = UNetWithScale(
            base_res=hyperparams.base_res,
            scale_dict={'scale' : 1.0},
            n_input_scalar_components = 1,
            n_input_vector_components = 0,
            n_output_scalar_components = 1,
            n_output_vector_components = 0,
            time_history = 1,
            time_future = 1,
            kernel_size = hyperparams.kernel_size,
            hidden_channels = hyperparams.hidden_channels,
            activation = 'gelu',
            norm = True,                                    
            ch_mults = hyperparams.ch_mults,
            is_attn = hyperparams.is_attn,
            mid_attn = False,
            n_blocks = 2,
            use1x1 = False,
            num_groups = hyperparams.num_groups,
            positional_embedding=hyperparams.positional_embedding
        ).to(device)

        model = UNet_kernel_interpolate(model, hyperparams.interpolation_mode, device=device)
    elif model_name.lower() == 'oformer':
        hyperparams = config.oformer

        model = OFormer(
            n_dim=hyperparams.n_dim,
            in_channels=hyperparams.in_channels,
            out_channels=hyperparams.out_channels,
            encoder_hidden_channels=hyperparams.encoder_hidden_channels,
            use_decoder=hyperparams.use_decoder,
            encoder_num_heads=hyperparams.encoder_num_heads,
            encoder_n_layers=hyperparams.encoder_n_layers,
            spectral_conv_encoder=hyperparams.spectral_conv_encoder,
            num_modes=hyperparams.num_modes,
            query_basis=hyperparams.query_basis,
            norm=hyperparams.norm
        )
        model = OFormerWrapper(model).to(device)
    else:
        model = get_model(config).to(device)
    
    print("Model params:", count_model_params(model))

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )

    if config.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.gamma,
            patience=config.opt.scheduler_patience,
            mode="min",
        )
    elif config.opt.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.scheduler_T_max
        )
    elif config.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")

    # Creating the losses
    train_loss = LpLossAbs(d=2, p=2)
    rel_l2loss = LpLoss(d=2, p=2)
    
    eval_losses = {"l2": rel_l2loss}

    if config.verbose:
        print("\n### MODEL ###\n", model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {train_loss}")
        print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

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

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    torch.save(model, os.path.join('./ckpts', save_name))