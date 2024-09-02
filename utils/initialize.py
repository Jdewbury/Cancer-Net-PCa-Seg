import monai
import torch
from .mamba_unet import LightMUNet

def get_model(args):
    """Retrieves and initializes select model architecture.

    Args:
        args: An object containing configuration parameters.

    Returns:
        Initialization of model.
    """
    if args.model == 'segresnet':
        return monai.networks.nets.SegResNet(
            spatial_dims=2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=args.init_filters,
            in_channels=1,
            out_channels=1,
            dropout_prob=0.2,
        )
    elif args.model =='unet':
        return monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif args.model == 'swinunetr':
        return monai.networks.nets.SwinUNETR(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            img_size=(args.size, args.size)
        )
    elif args.model == 'attentionunet':
        return monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
    elif args.model == 'mambaunet':
        return LightMUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            init_filters=args.init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1)
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}. Choose from 'segresnet', 'unet', 'swinunetr', 'attentionunet', or 'mambaunet'.")

def get_optimizer(args, model):
    """Retrieves and initializes select optimizer.

    Args:
        args: An object containing configuration parameters.
        model: The model whose parameters will be optimized.

    Returns:
        Initialization of optimizer.

    Raises:
        ValueError: If an invalid optimizer name is provided.
    """
    if args.optimizer == 'adam':
        print('Using Adam optimizer')
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    elif args.optimizer == 'adamw':
        print('Using AdamW optimizer')
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    elif args.optimizer == 'sgd':
        print('Using SGD optimizer')
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Invalid optimizer name: {args.optimizer}. Choose from 'adam', 'adamw', or 'sgd'.")

def get_scheduler(args, optimizer):
    """Retrieves and initializes select learning rate scheduler.

    Args:
        args: An object containing configuration parameters.
        optimizer: The optimizer whose learning rate will be scheduled.

    Returns:
        Initialization of learning rate scheduler, or None if no scheduler is specified.
    """
    if args.scheduler == 'step':
        print('Using StepLR')
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs // 4), gamma=args.lr_step)
    elif args.scheduler == 'cosine':
        print('Using CosineAnnealingLR')
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs // 4), eta_min=0)
    else:
        print('No LR scheduler')
        return None