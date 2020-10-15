import torch
import torch.nn as nn
import torchvision
from datetime import datetime
import models.DCGAN as DCGAN
import os
from collections import OrderedDict

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')