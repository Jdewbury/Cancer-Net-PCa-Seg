import argparse
import torch
from torchvision import transforms
import numpy as np
import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from data_utils import list_nii_paths, list_prostate_paths
from dataset import CancerNetPCa

import os
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for the training and validation loops.')
parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs.')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate for training.')
parser.add_argument('--model', default='unet', type=str, help='Model architecture to be used for training.')
parser.add_argument('--img_dir', default='data/images', type=str, help='Directory containing image data.')
parser.add_argument('--mask_dir', default='data_2', type=str, help='Directory containing mask data.')
parser.add_argument('--prostate_mask', action='store_true', help='Flag to use prostate mask.')
parser.add_argument('--size', default=256, help='Desired size of image and mask.')
parser.add_argument('--slice', default=9, help='Slice to be evaluated.')
parser.add_argument('--val_interval', default=2, type=int, help='Epoch interval for evaluation on validation set.')
parser.add_argument('--save', action='store_true', help='Save best model weights.')
parser.add_argument('--test', action='store_true', help='Evaluate model on test set.')

args = parser.parse_args()

if args.model == 'segresnet':
    print('Using SegResNet')
    model = monai.networks.nets.SegResNet(
        spatial_dims=2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=1,
        dropout_prob=0.2,
    )

if args.model == 'unet':
    print('Using UNet')
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs // 4), gamma=0.1)

img_paths = list_nii_paths(args.img_dir)
mask_paths = list_prostate_paths(args.mask_dir)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img / img.max() if img.max() > 0 else img)
])

dataset = CancerNetPCa(img_path=img_paths, mask_path=mask_paths, batch_size=args.batch_size, img_size=(args.size, args.size), 
                       slice_num=args.slice, prostate=args.prostate_mask, transform=transform)

loss_function = DiceLoss(sigmoid=True)
dice_metric = DiceMetric(include_background=True, reduction='mean')

dir = f'{args.prostate_mask*"prostate-"}{args.model}'

if args.save:
    count = 1
    unique_dir = f"scores/{dir}_{count}"
    
    while os.path.exists(unique_dir):
        count += 1
        unique_dir = f"scores/{dir}_{count}"
    
    weight_dir = f'models/{dir}_{count}'
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

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0
    for batch_data in dataset.train:
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        outputs = torch.sigmoid(outputs)
        dice_metric(y_pred=outputs, y=labels)

    train_metric = dice_metric.aggregate().item()
    dice_metric.reset()

    train_loss.append(epoch_loss)
    train_dice.append(train_metric)

    scheduler.step()
    print(f'epoch {epoch + 1}, learning rate: {scheduler.get_last_lr()[0]:.1e}, train loss: {epoch_loss:.4f}, train dice: {train_metric:.4f}')

    if (epoch + 1) % int(args.val_interval) == 0:
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for val_data in dataset.val:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)

                loss = loss_function(val_outputs, val_labels)
                epoch_loss += loss.item()
                val_outputs = torch.sigmoid(val_outputs)
                dice_metric(y_pred=val_outputs, y=val_labels)

            val_metric = dice_metric.aggregate().item()
            dice_metric.reset()

            val_loss.append(epoch_loss)
            val_dice.append(val_metric)

            if args.save and val_metric > best_metric:
                best_metric = val_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), weight_path)
                no_improvement = 0
                print(f"Saving new best model, best metric: {best_metric} at epoch: {best_metric_epoch}")

print(f'Training completed, best metric: {best_metric} at epoch: {best_metric_epoch} saved at: {weight_path}')

if args.test:
    print('Starting Testing')
    model.load_state_dict(torch.load(weight_path))
    start_time = perf_counter()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for test_data in dataset.test:
            test_inputs, test_labels = test_data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)

            loss = loss_function(test_outputs, test_labels)
            test_loss += loss.item()
            test_outputs = torch.sigmoid(test_outputs)
            dice_metric(y_pred=test_outputs, y=test_labels)

        test_dice = dice_metric.aggregate().item()
    
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    print(f'test loss: {test_loss:.4f}, test dice: {test_dice:.4f}, total time: {elapsed_time}')

if args.save:
    print(f'Saving values at {unique_dir}')

    np.save(f'{unique_dir}/train_dice.npy', train_dice)
    np.save(f'{unique_dir}/train_loss.npy', train_loss)
    np.save(f'{unique_dir}/val_dice.npy', val_dice)
    np.save(f'{unique_dir}/val_loss.npy', val_loss)
    np.save(f'{unique_dir}/test_dice.npy', test_dice)
    np.save(f'{unique_dir}/test_loss.npy', test_loss)



