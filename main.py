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
CONFIG.epochs = 10
CONFIG.batch_size = 64
CONFIG.img_size = 224
CONFIG.hidden_size = 512
CONFIG.model_dim = 512
CONFIG.output_dim = 64
CONFIG.num_heads = 8
CONFIG.num_layers = 8
CONFIG.patch_size = 16
CONFIG.weight_decay = 0.1
CONFIG.ckpt_path = "./checkpoints/checkpoint.pt"
CONFIG.data_dir = "./data/testing"

train(CONFIG, True)