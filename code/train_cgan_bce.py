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

MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = 'cgan-model-bce-epoch10.pth'
EPOCH = 200
SAVE_INTERVAL = 20
# for generation
SAMPLE_INTERVAL = 469
SAMPLE_SIZE = 64
# shape of a single image
INPUT_IMG_SHAPE = (1, 28, 28)

# Hyperparameters
BATCH_SIZE = 128
Z_DIM = 100
GENERATOR_LR = 0.0002
DISCRIMINATOR_LR = 0.0002
BETAS = (0.5, 0.999)

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

    # input image sample
    # data_iter = iter(trainset)
    # img, label = next(data_iter)
    # torchvision.utils.save_image(img, 'sample.png')

    # 2. instantiate the model
    # Generator G
    INPUT_DIM = Z_DIM + num_classes 
    # we will append a one-hot vector to the z_dim
    # where the one-hot vector is of num_classes dimension
    G = DCGAN.Generator(
        channels=[INPUT_DIM, 256, 128, 64, 1],
        kernel_sizes=[None, 7, 5, 4, 4],
        strides=[None, 1, 1, 2, 2],
        paddings=[None, 0, 2, 1, 1],
        batch_norm=True,
        activations=[nn.ReLU(), nn.Tanh()]
    )

    # Discriminator D
    INPUT_CHANNEL = INPUT_IMG_SHAPE[0] + num_classes
    # we will append (batch_size, num_classes, 28, 28), which is similar to the one-hot vector
    # to (batch_size, 1, 28, 28) which is the input batch of MNIST
    # for example, if dimension 3 of the one-hot vector has the value 1, then we create a tensor of size (1, 28, 28) full on 1s for index 3
    # and the other 9 tensor will be filled with 0s
    D = DCGAN.Discriminator(
        channels=[INPUT_CHANNEL, 64, 128, 256, 1],
        kernel_sizes=[None, 4, 4, 5, 7],
        strides=[None, 2, 2, 1, 1],
        paddings=[None, 1, 1, 2, 0],
        batch_norm=False,
        activation=nn.LeakyReLU(0.2),
    )

    print(f"""
Generator G:
{G}
Critic D:
{D}
""", flush=True)

    # setting for training using multiple GPUs
    multigpu = False
    if torch.cuda.device_count() > 1:
        print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        multigpu = True
    
    # move the models to the device that we use for training
    G.to(device)
    D.to(device)

    # 3. define the loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. define the optimisers
    g_optim = optim.Adam(G.parameters(), lr=GENERATOR_LR, betas=BETAS)
    d_optim = optim.Adam(D.parameters(), lr=DISCRIMINATOR_LR, betas=BETAS)

    # 5. train the model
    MODEL_DIRPATH = MAIN_DIR + 'saved_models/'
    GENERATED_DIRPATH = MAIN_DIR + 'generated_images/'
    SAVED_MODEL_NAME = MODEL_DIRPATH + CONTINUE_TRAIN_NAME

    next_epoch = 0
    if CONTINUE_TRAIN:
        checkpoint = torch.load(SAVED_MODEL_NAME)
        G.load_state_dict(checkpoint.get('G_state_dict'))
        D.load_state_dict(checkpoint.get('D_state_dict'))
        g_optim.load_state_dict(checkpoint.get('g_optim_state_dict'))
        d_optim.load_state_dict(checkpoint.get('d_optim_state_dict'))
        next_epoch = checkpoint.get('epoch')
    
    def save_training_progress(G, D, g_optimizer, d_optimizer, epoch, target_dir):
        torch.save({
            'epoch' : epoch + 1,
            'G_state_dict' : G.state_dict(),
            'D_state_dict' : D.state_dict(),
            'g_optim_state_dict' : g_optimizer.state_dict(),
            'd_optim_state_dict' : d_optimizer.state_dict()
        }, target_dir)