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

if __name__ == "__main__":
    
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}\n", flush=True)

    # directory setup
    if 'data' not in os.listdir(MAIN_DIR):
        print('creating data directory...', flush=True)
        os.mkdir(MAIN_DIR + 'data')
    if 'generated_images' not in os.listdir(MAIN_DIR):
        print('creating generated_images directory...', flush=True)
        os.mkdir(MAIN_DIR + 'generated_images')
    if 'logs' not in os.listdir(MAIN_DIR):
        print('creating logs directory...', flush=True)
        os.mkdir(MAIN_DIR + 'logs')
    if 'saved_models' not in os.listdir(MAIN_DIR):
        print('creating saved_models directory...', flush=True)
        os.mkdir(MAIN_DIR + 'saved_models')
    
    # 1. load the dataset
    DATA_PATH = MAIN_DIR + 'data/'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])

    trainset = datasets.MNIST(root=DATA_PATH, download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # total unique classes in MNIST dataset
    num_classes = len(trainset.classes)

    print(f"""
Total training data: {len(trainset)}
Total unique classes: {num_classes}
    """, flush=True)