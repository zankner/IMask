import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models import vgg19
import ml_collections
from model_utils.train import train
import matplotlib as plt

CONFIG = ml_collections.ConfigDict()
CONFIG.epochs = 100
CONFIG.batch_size = 32
CONFIG.img_size = 224
CONFIG.hidden_size = 64
CONFIG.model_dim = 64
CONFIG.num_heads = 8
CONFIG.num_layers = 8
CONFIG.patch_size = 16

train(CONFIG)