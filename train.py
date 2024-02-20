import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='CancerNet-PCa Training')
parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--prostate_mask", default=False, type=bool, help="Base learning rate at the start of the training.")

args = parser.parse_args()