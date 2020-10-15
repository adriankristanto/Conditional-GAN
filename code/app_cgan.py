import torch
import torch.nn as nn
import torchvision
from datetime import datetime
import models.DCGAN as DCGAN
import os
from collections import OrderedDict

# path to the pth file
MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
MODEL_DIR = MAIN_DIR + 'saved_models/'
FILE_PATH = MODEL_DIR + 'cgan-model-epoch100.pth'

Z_DIM = 100
num_classes = 10
INPUT_DIM = Z_DIM + num_classes

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
G = DCGAN.Generator(
    channels=[INPUT_DIM, 256, 128, 64, 1],
    kernel_sizes=[None, 7, 5, 4, 4],
    strides=[None, 1, 1, 2, 2],
    paddings=[None, 0, 2, 1, 1],
    batch_norm=True,
    activations=[nn.ReLU(), nn.Tanh()]
)