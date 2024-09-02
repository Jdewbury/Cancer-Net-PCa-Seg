import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from monai.metrics import DiceMetric
from utils.data_utils import list_nii_paths, list_prostate_paths
from dataset import CancerNetPCa
from utils.initialize import get_model
from thop import profile

import os
from time import perf_counter
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='unet', type=str, help='Model architecture that was used for training.')
parser.add_argument('--img_dir', default='data/images', type=str, help='Directory containing image data.')
parser.add_argument('--mask_dir', default='data_2', type=str, help='Directory containing mask data.')
parser.add_argument('--weight_dir', default='models', type=str, help='Directory containing model weight(s).')
parser.add_argument('--param_dir', default='scores', type=str, help='Directory containing model parameters.')
parser.add_argument('--params', action='store_true', help='Print total number of model parameters and FLOPs.')
parser.add_argument('--save', action='store_true', help='Save inference results.')

args = parser.parse_args()
args_dict = vars(args)

model_loss = []
model_dice = []
inference_time = []

if args.save:
    save_dir = f'evaluate/{args.model}'
    os.makedirs(save_dir, exist_ok=True)

# check if evaluating single model, or set of models
if args.model.count('-') > 0:
    model_name = args.model.split('-')[0]
else:
    model_name = args.model

for score in os.listdir(args.param_dir):
    if score.startswith(args.model):
        saved_params = np.load(f'{args.param_dir}/{score}/params.npy', allow_pickle=True).tolist()
        saved_args = SimpleNamespace(**saved_params)
        model_weights = torch.load(f'{args.weight_dir}/{score}/CancerNetPCa.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        model = get_model(saved_args)

        img_paths = list_nii_paths(saved_args.img_dir)
        mask_paths = list_prostate_paths(saved_args.mask_dir)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((saved_args.size, saved_args.size)),
            transforms.ToTensor(),
        ])

        dataset = CancerNetPCa(img_path=img_paths, mask_path=mask_paths, seed=saved_args.seed, batch_size=saved_args.batch_size,
                                prostate=saved_args.prostate_mask, transform=transform)

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
        total_time = elapsed_time / (len(dataset.test)*saved_args.batch_size)

        print(f'Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, Average time: {total_time:.4f}')
        model_loss.append(test_loss)
        model_dice.append(test_dice)
        
        inference_time.append(total_time)

        if args.params:
            input_tensor = torch.randn(saved_args.batch_size, 1, saved_args.size, saved_args.size).to(device)
            flops, params = profile(model, inputs=(input_tensor,))
            print(f'Total GFLOPs: {flops/10**9}')
            print(f'Total M parameters: {params/10**6}')

scores = {
    'inference-time': inference_time,
    'test-loss': model_loss,
    'test-dice': model_dice,
}

if args.save:
    print(f'Saving values at {save_dir}')
    np.save(f'{save_dir}/scores.npy', scores)



