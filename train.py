import argparse
import torch
from torchvision import transforms
import numpy as np
import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from data_utils import list_nii_paths, list_prostate_paths
from dataset import CancerNetPCa

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
parser.add_argument('--save', action='store_true', help='Save best model weights.')
parser.add_argument('--val_interval', default=2, type=int, help='Save best model weights.')

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

if args.prostate_mask:
    weight_path = f'models/CancerNetPCa-prostate-{args.model}.pth'
else:
    weight_path = f'models/CancerNetPCa-{args.model}.pth'

print('Starting Training')

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
model = model.to(device)

best_metric = 0
best_metric_epoch = 0

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
    print(f'epoch {epoch + 1} average loss: {epoch_loss:.4f}')

    if (epoch + 1) % int(args.val_interval) == 0:
        model.eval()
        with torch.no_grad():
            val_dice = []
            for val_data in dataset.val:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)

                val_outputs = torch.sigmoid(val_outputs)
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            val_dice.append(metric)
            dice_metric.reset()
            mean_val_dice = np.mean([score for score in val_dice if not np.isnan(score)])
            if args.save and mean_val_dice > best_metric:
                best_metric = mean_val_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), weight_path)
                no_improvement = 0
                print(f"Best metric: {best_metric} at epoch: {best_metric_epoch}")

print(f'Training completed, best metric: {best_metric} at epoch: {best_metric_epoch} saved at {weight_path}')


