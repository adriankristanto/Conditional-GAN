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

# loading a model that was wrapped by nn.DataParallel for training
checkpoint = torch.load(FILE_PATH, map_location=device)
old_G_state_dict = checkpoint.get('G_state_dict')
# if the model was wrapped by nn.DataParallel
if 'module.' in list(old_G_state_dict.keys())[0]:
    new_G_state_dict = OrderedDict()
    for key, value in old_G_state_dict.items():
        # remove "module." from each key
        name = key[7:]
        new_G_state_dict[name] = value
    # load the newly created state dict
    G.load_state_dict(new_G_state_dict)
else:
    G.load_state_dict(old_G_state_dict)