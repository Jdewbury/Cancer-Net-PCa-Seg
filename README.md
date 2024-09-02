# Cancer-Net PCa-Seg: Benchmarking Deep Learning Models for Prostate Cancer Segmentation Using Synthetic Correlated Diffusion Imaging

Prostate cancer (PCa) is the most prevalent cancer among men in the United States, accounting for nearly 300,000 cases, 29\% of all diagnoses and 35,000 total deaths in 2024. Traditional screening methods such as prostate-specific antigen (PSA) testing and magnetic resonance imaging (MRI) have been pivotal in diagnosis, but have faced limitations in specificity and generalizability. In this paper, we explore the potential of enhancing PCa lesion segmentation using a novel MRI modality called synthetic correlated diffusion imaging (CDI<sup>s</sup>). We employ several state-of-the-art deep learning models, including U-Net, SegResNet, Swin UNETR, Attention U-Net, and LightM-UNet, to segment PCa lesions from a 200 CDI<sup>s</sup> patient cohort. We find that SegResNet achieved superior segmentation performance with a Dice-Sørensen coefficient (DSC) of $76.68 \pm 0.8$. Notably, the Attention U-Net, while slightly less accurate (DSC $74.82 \pm 2.0$), offered a favorable balance between accuracy and computational efficiency. Our findings demonstrate the potential of deep learning models in improving PCa lesion segmentation using CDI<sup>s</sup> to enhance PCa management and clinical support. 
<br><br>
This repository contains modules and instructions for replicating and extending experiments featured in our paper:
- A training script `./train.py` to train the select architectures on the PCa CDI<sup>s</sup> data
- An inference script `./evaluate.py` to evaluate the trained networks on the testing data

## Dataset
Cancer-Net PCa-Data is an open access benchmark dataset of volumetric correlated diffusion imaging (CDIs) data acquisitions of prostate cancer patients. Cancer-Net PCa-Data is a part of the Cancer-Net open source initiative dedicated to advancement in machine learning and imaging research to aid clinicians in the global fight against cancer.

The volumetric CDIs data acquisitions in the Cancer-Net PCa-Data dataset were generated from a patient cohort of 200 patient cases acquired at Radboud University Medical Centre (Radboudumc) in the Prostate MRI Reference Center in Nijmegen, The Netherlands and made available as part of the SPIE-AAPM-NCI PROSTATEx Challenges. Masks derived from the PROSTATEx_masks repository are also provided which label regions of healthy prostate tissue, clinically significant prostate cancer (csPCa), and clinically insignificant prostate cancer (insPCa).

This [dataset](https://www.kaggle.com/datasets/hgunraj/cancer-net-pca-data) is being used to train and validate our Cancer-Net PCa-Seg models for PCa lesion segmentation from CDIs data acquisitions.

## Train
If you want to train a network from scratch, here is a sample training run:
```
python3 train.py --model MODEL --img_dir IMG_DIR --mask_dir MASK_DIR
```
Full CLI:
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--seed SEED] [--learning_rate LEARNING_RATE] [--model MODEL] [--img_dir IMG_DIR]
                [--mask_dir MASK_DIR] [--prostate_mask] [--size SIZE] [--val_interval VAL_INTERVAL] [--lr_step LR_STEP] [--scheduler SCHEDULER]
                [--optimizer OPTIMIZER] [--weights WEIGHTS] [--init_filters INIT_FILTERS] [--save]

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size for the training and validation loops.
  --epochs EPOCHS       Total number of training epochs.
  --seed SEED           Seed to use for splitting dataset.
  --learning_rate LEARNING_RATE
                        Initial learning rate for training.
  --model MODEL         Model architecture to be used for training.
  --img_dir IMG_DIR     Directory containing image data.
  --mask_dir MASK_DIR   Directory containing mask data.
  --prostate_mask       Flag to use prostate mask.
  --size SIZE           Desired size of image and mask.
  --val_interval VAL_INTERVAL
                        Epoch interval for evaluation on validation set.
  --lr_step LR_STEP     Epoch interval for evaluation on validation set.
  --scheduler SCHEDULER
                        Learning rate scheduler to use.
  --optimizer OPTIMIZER
                        Learning rate scheduler to use.
  --weights WEIGHTS     Path to pretrained model weights to use.
  --init_filters INIT_FILTERS
                        Number of filters for model.
  --save                Save results.
```
## Inference
If you want to run inference on your trained model, here is a sample run:
```
python3 evaluate.py --model MODEL --img_dir IMG_DIR --mask_dir MASK_DIR
```
Full CLI:
```
usage: evaluate.py [-h] [--model MODEL] [--img_dir IMG_DIR] [--mask_dir MASK_DIR] [--weight_dir WEIGHT_DIR] [--param_dir PARAM_DIR] [--params] [--save]

options:
  -h, --help            show this help message and exit
  --model MODEL         Model architecture that was used for training.
  --img_dir IMG_DIR     Directory containing image data.
  --mask_dir MASK_DIR   Directory containing mask data.
  --weight_dir WEIGHT_DIR
                        Directory containing model weight(s).
  --param_dir PARAM_DIR
                        Directory containing model parameters.
  --params              Print total number of model parameters and FLOPs.
  --save                Save inference results.
```
