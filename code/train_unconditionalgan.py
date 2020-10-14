import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from tqdm import tqdm 
import models.DCGAN as DCGAN
import os 
from utils import CriticLoss, GradientPenaltyLoss, GeneratorLoss

MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = 'unconditionalgan-model-epoch10.pth'
EPOCH = 200
SAVE_INTERVAL = 20
# for generation
SAMPLE_INTERVAL = 100
SAMPLE_SIZE = 32
# shape of a single image
INPUT_IMG_SHAPE = (1, 28, 28)

# Hyperparameters
BATCH_SIZE = 128
Z_DIM = 100
GENERATOR_LR = 0.0002
DISCRIMINATOR_LR = 0.0002
BETAS = (0.5, 0.999)

# WGAN-GP global variables
# the recommended ratio used is 5 critic updates to 1 generator update
CRITIC_ITER = 5
# the following is the penalty coefficient
LAMBDA = 10