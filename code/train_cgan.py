import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import models.DCGAN as DCGAN
import os

# Hyperparameters
BATCH_SIZE = 128

if __name__ == "__main__":
    
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}\n", flush=True)

    # directory setup
    MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
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
    DATA_PATH = MAIN_DIR + 'data'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])

    trainset = datasets.MNIST(root=DATA_PATH, download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Total training data: {len(trainset)}\n", flush=True)