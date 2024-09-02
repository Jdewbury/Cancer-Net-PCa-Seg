import monai
from .mamba_unet import LightMUNet

def get_model(model, init_filters=32, size=256):
    print(model, model== 'unet')
    if model == 'segresnet':
        return monai.networks.nets.SegResNet(
            spatial_dims=2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=init_filters,
            in_channels=1,
            out_channels=1,
            dropout_prob=0.2,
        )
    elif model =='unet':
        return monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model == 'swinunetr':
        return monai.networks.nets.SwinUNETR(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            img_size=(size, size)
        )
    elif model == 'attentionunet':
        return monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
    elif model == 'mambaunet':
        return LightMUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1)
        )
    else:
        raise ValueError(f"Invalid model name: {model}. Choose from 'segresnet', 'unet', 'swinunetr', 'attentionunet', or 'mambaunet'.")