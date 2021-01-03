import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:3])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x, y, masked_patch):
        height_offset = 16 * ((masked_patch) // 2)
        width_offset = 16 * ((masked_patch) % 2)
        y = y[:, :, height_offset:height_offset + 16,
              width_offset:width_offset + 16]

        y = (y - self.mean) / self.std
        y = self.feature_extractor(y)
        y = F.avg_pool2d(y, 16, 1)
        y = y.view(1, 64)

        x = x[:, masked_patch, :]

        loss = F.mse_loss(x, y)
        return loss