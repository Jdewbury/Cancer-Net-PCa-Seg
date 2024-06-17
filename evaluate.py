import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from data_utils import list_nii_paths, list_prostate_paths
from dataset import CancerNetPCa
from mamba_unet import LightMUNet
from thop import profile

import os
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='unet', type=str, help='Model architecture to be used for training.')
parser.add_argument('--img_dir', default='data/images', type=str, help='Directory containing image data.')
parser.add_argument('--mask_dir', default='data_2', type=str, help='Directory containing mask data.')
parser.add_argument('--weight_dir', default='models', type=str, help='Directory containing model weight(s).')
parser.add_argument('--param_dir', default='scores', type=str, help='Directory containing model parameters.')
parser.add_argument('--save', action='store_true', help='Save inference results.')
parser.add_argument('--params', action='store_true', help='Print total number of model parameters and FLOPs.')

args = parser.parse_args()
args_dict = vars(args)

model_loss = []
model_dice = []
time = []

if args.save:
    dir = f'evaluate/{args.model}'
    os.mkdir(dir)

# check if using single weight, or group of weights
if args.model.count('-') > 0:
    model = args.model.split('-')[0]
else:
    model = args.model

for score in os.listdir(args.param_dir):
    if score.startswith(args.model):
        params = np.load(f'{args.param_dir}/{score}/params.npy', allow_pickle=True).tolist()
        model_weights = torch.load(f'{args.weight_dir}/{score}/CancerNetPCa.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        batch_size = params['batch_size']
        seed = params['seed']
        learning_rate = params['learning_rate']
        size = params['size']
        init_filters = params['init_filters']
        if params['prostate_mask']:
            prostate_mask = True
        else:
            prostate_mask = False

        if model == 'segresnet':
            model = monai.networks.nets.SegResNet(
                spatial_dims=2,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=init_filters,
                in_channels=1,
                out_channels=1,
                dropout_prob=0.2,
            )
        if model == 'unet':
            model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        if model == 'swinunetr':
            model = monai.networks.nets.SwinUNETR(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                img_size=(size, size)
            )
        if model == 'attentionunet':
            model = monai.networks.nets.AttentionUnet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2)
            )
        if model == 'mambaunet':
            model = LightMUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                init_filters=init_filters,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1)
            )

        img_paths = list_nii_paths(args.img_dir)
        mask_paths = list_prostate_paths(args.mask_dir)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        dataset = CancerNetPCa(img_path=img_paths, mask_path=mask_paths, seed=seed, batch_size=batch_size,
                                prostate=prostate_mask, transform=transform)
        
        print(len(dataset.test)*batch_size)

        #loss_seg = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        loss_ce = nn.BCEWithLogitsLoss(reduction='mean')
        dice_metric = DiceMetric(include_background=True, reduction='mean', get_not_nans=False)

        device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(model_weights)
        model = model.to(device)

        elapsed_time = 0

        print('Starting Testing for', score)
        model.eval()

        start_time = perf_counter()
        with torch.no_grad():
            test_loss = 0

            for step, test_data in enumerate(dataset.test):
                test_inputs, test_labels = test_data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)

                #loss = loss_seg(test_outputs, test_labels) + loss_ce(test_outputs, test_labels.float())
                loss = loss_ce(test_outputs, test_labels)
                test_loss += loss.item()
                test_outputs = torch.sigmoid(test_outputs)
                test_outputs_norm = test_outputs / test_outputs.max()
                test_outputs_binary = (test_outputs_norm > 0.5).float()
                dice_metric(y_pred=test_outputs_binary, y=test_labels)

            test_dice = dice_metric.aggregate().item()
        
        test_loss /= step

        end_time = perf_counter()
        elapsed_time += (end_time - start_time)
        total_time = elapsed_time / (len(dataset.test)*batch_size)

        print(f'Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, Average time: {total_time:.4f}')
        model_loss.append(test_loss)
        model_dice.append(test_dice)
        
        time.append(total_time)

scores = {
    'inference-time': time,
    'test-loss': model_loss,
    'test-dice': model_dice,
}

if args.save:
    print(f'Saving values at {dir}')
    np.save(f'{dir}/scores.npy', scores)

if args.params:
    input_tensor = torch.randn(batch_size, 1, size, size).to(device)

    flops, params = profile(model, inputs=(input_tensor,))

    print(f'Total GFLOPs: {flops/10**9}')
    print(f'Total M parameters: {params/10**6}')



