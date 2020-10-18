import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datetime import datetime
import models.DCGAN as DCGAN
import os
from collections import OrderedDict

# path to the pth file
MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
MODEL_DIR = MAIN_DIR + 'saved_models/'
FILE_PATH = MODEL_DIR + 'cgan-model-bce-epoch180_32x32.pth'

SAMPLE_SIZE = 1024

Z_DIM = 100
num_classes = 10
INPUT_DIM = Z_DIM + num_classes

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
G = DCGAN.Generator(
    channels=[INPUT_DIM, 512, 256, 128, 1],
    kernel_sizes=[None, 4, 4, 4, 4],
    strides=[None, 1, 2, 2, 2],
    paddings=[None, 0, 1, 1, 1],
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

# accepts user input
# reference: https://stackoverflow.com/questions/56513576/converting-tensor-to-one-hot-encoded-tensor-of-indices/56523313
# to convert a tensor to an index tensor, use .long() to convert it to int64
digit = torch.Tensor(
    [int(input("Choose a digit (0-9): "))]
    ).long().to(device)

# encode the index tensor from the user input
# repeat it SAMPLE_SIZE times for the batch dimension to make it to shape (SAMPLE_SIZE, num_classes)
# finally, expand the dim to have 4 dimensions with [:, :, None, None]
one_hot_digit = F.one_hot(digit, num_classes=num_classes).repeat((SAMPLE_SIZE, 1))[:, :, None, None]

# create a noise vector
noise = torch.randn((SAMPLE_SIZE, Z_DIM, 1, 1)).to(device)

# concatenate the noise with the one hot vector
noise_onehot = torch.cat([noise.float(), one_hot_digit.float()], dim=1)

# generate a new image
images = G(noise_onehot)

# save the generated images
torchvision.utils.save_image(images, datetime.now().strftime('%d_%m_%Y_%H%M%S') + '.png', nrow=32)