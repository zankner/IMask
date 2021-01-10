import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self,
                 patch_size,
                 output_dim,
                 pool_size,
                 img_size,
                 device="cpu"):
        super(PerceptualLoss, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.num_patch_axis = img_size // patch_size

        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:7])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1,
                                                             1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1,
                                                            1).to(device)

    def forward(self, x, y, masked_patches):
        y = (y - self.mean) / self.std

        masked_patches_dims = []
        for masked_patch in masked_patches:
            height_offset = self.patch_size * (
                (masked_patch) // self.num_patch_axis)
            width_offset = self.patch_size * (
                (masked_patch) % self.num_patch_axis)
            masked_patches_dims.append([height_offset, width_offset])

        target_patches = []
        for target in range(y.shape[0]):
            for masked_patch in masked_patches_dims:
                height_offset, width_offset = masked_patch
                extracted_patch = y[target, :, height_offset:height_offset +
                                    self.patch_size,
                                    width_offset:width_offset +
                                    self.patch_size]
                target_patches.append(extracted_patch)

        target_patches_tensor = torch.stack(target_patches)
        target_patches_tensor = self.feature_extractor(target_patches_tensor)
        target_patches_tensor = F.avg_pool2d(target_patches_tensor,
                                             self.pool_size, 1)
        target_patches_tensor = target_patches_tensor.view(
            target_patches_tensor.shape[0], self.output_dim)

        masked_patches_shifted = [
            masked_patch + 1 for masked_patch in masked_patches
        ]
        x = x[:, masked_patches_shifted, :]
        x = x.view(target_patches_tensor.shape[0], self.output_dim)

        loss = F.mse_loss(x, target_patches_tensor)
        return loss