import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from monai.metrics import DiceMetric
from utils.data_utils import list_nii_paths, list_prostate_paths, load_weights
from dataset import CancerNetPCa
from utils.initialize import get_model, get_optimizer, get_scheduler

import os
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for the training and validation loops.')
parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs.')
parser.add_argument('--seed', default=42, type=int, help='Seed to use for splitting dataset.')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate for training.')
parser.add_argument('--model', default='unet', type=str, help='Model architecture to be used for training.')
parser.add_argument('--img_dir', default='data/images', type=str, help='Directory containing image data.')
parser.add_argument('--mask_dir', default='data_2', type=str, help='Directory containing mask data.')
parser.add_argument('--prostate_mask', action='store_true', help='Flag to use prostate mask.')
parser.add_argument('--size', default=128, type=int, help='Desired size of image and mask.')
parser.add_argument('--val_interval', default=2, type=int, help='Epoch interval for evaluation on validation set.')
parser.add_argument('--lr_step', default=0.1, type=float, help='Epoch interval for evaluation on validation set.')
parser.add_argument('--scheduler', default=None, type=str, help='Learning rate scheduler to use.')
parser.add_argument('--optimizer', default='adam', type=str, help='Learning rate scheduler to use.')
parser.add_argument('--weights', default=None, type=str, help='Path to pretrained model weights to use.')
parser.add_argument('--init_filters', default=32, type=int, help='Number of filters for model.')
parser.add_argument('--save', action='store_true', help='Save results.')
parser.add_argument('--test', action='store_true', help='Evaluate model on test set.')

args = parser.parse_args()
default_args = {action.dest: action.default for action in parser._actions if action.dest != 'help'}
args_dict = {**default_args, **vars(args)}

model = get_model(args)
optimizer = get_optimizer(args, model)
scheduler = get_scheduler(args, optimizer)

if args.weights:
    model = load_weights(model, args.weights)

img_paths = list_nii_paths(args.img_dir)
mask_paths = list_prostate_paths(args.mask_dir)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
])

dataset = CancerNetPCa(img_path=img_paths, mask_path=mask_paths, seed=args.seed, batch_size=args.batch_size,
                        prostate=args.prostate_mask, transform=transform)

loss_ce = nn.BCEWithLogitsLoss(reduction='mean')
dice_metric = DiceMetric(include_background=True, reduction='mean', get_not_nans=False)

dir = f'{args.prostate_mask*"pro-"}{args.model}'

if args.save:
    count = 1
    unique_dir = f'scores/{dir}-{count}'
    
    while os.path.exists(unique_dir):
        count += 1
        unique_dir = f'scores/{dir}-{count}'
    
    weight_dir = f'models/{dir}-{count}'
    weight_path = f'{weight_dir}/CancerNetPCa.pth'
    os.mkdir(unique_dir)
    os.mkdir(weight_dir)

print('Starting Training')

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
model = model.to(device)

best_metric = 0
best_metric_epoch = 0
train_loss = []
train_dice = []
val_loss = []
val_dice = []

elapsed_time = 0

for epoch in range(args.epochs):
    start_time = perf_counter()
    model.train()
    epoch_loss = 0

    for step, batch_data in enumerate(dataset.train):
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_ce(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        outputs = torch.sigmoid(outputs)
        outputs_norm = outputs / outputs.max()
        outputs_binary = (outputs_norm > 0.5).float()
        dice_metric(y_pred=outputs_binary, y=labels)

    train_metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    epoch_loss /= step
    train_loss.append(epoch_loss)
    train_dice.append(train_metric)

    if scheduler:
        scheduler.step()
        print(f'epoch {epoch + 1}, learning rate: {scheduler.get_last_lr()[0]:.1e}, train loss: {epoch_loss:.4f}, train dice: {train_metric:.4f}')
    else:
        print(f'epoch {epoch + 1}, learning rate: {args.learning_rate}, train loss: {epoch_loss:.4f}, train dice: {train_metric:.4f}')

    if (epoch + 1) % int(args.val_interval) == 0:
        model.eval()

        with torch.no_grad():
            epoch_loss = 0

            for step, val_data in enumerate(dataset.val):
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)

                loss = loss_ce(val_outputs, val_labels)
                epoch_loss += loss.item()
                val_outputs = torch.sigmoid(val_outputs)
                val_outputs_norm = val_outputs / val_outputs.max()
                val_outputs_binary = (val_outputs_norm > 0.5).float()
                dice_metric(y_pred=val_outputs_binary, y=val_labels)

            val_metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            epoch_loss /= step
            val_loss.append(epoch_loss)
            val_dice.append(val_metric)
                
            if val_metric > best_metric:
                best_metric = val_metric
                best_metric_epoch = epoch + 1
                if args.save:
                    print(f'Saving new best model, best metric loss: {epoch_loss}, with dice: {val_metric} at epoch: {best_metric_epoch}')
                    torch.save(model.state_dict(), weight_path)
                else:
                    print(f'Best metric loss: {epoch_loss}, with dice: {val_metric} at epoch: {best_metric_epoch}')
                    
                no_improvement = 0

    end_time = perf_counter()
    elapsed_time += (end_time - start_time)

print(f'Training completed, best metric loss: {epoch_loss}, with dice: {val_metric} at epoch: {best_metric_epoch}, total train time: {elapsed_time}')

scores = {
    'time': elapsed_time,
    'train-loss': train_loss,
    'train-dice': train_dice,
    'val-loss': val_loss,
    'val-dice': val_dice
}

if args.save:
    print(f'Saving values at {unique_dir}')
    np.save(f'{unique_dir}/scores.npy', scores)
    np.save(f'{unique_dir}/params.npy', args_dict)



