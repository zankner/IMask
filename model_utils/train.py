import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model_utils.vision_transformer import VisionTransformer
from model_utils.perceptual_loss import PerceptualLoss


def _train_step(model, loss_fn, device, train_loader, optimizer, num_patches,
                epoch):
    patches = list(range(num_patches))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        np.random.shuffle(patches)
        masked_patches = patches[:len(patches) // 2]
        unmasked = [patch for patch in patches if patch not in masked_patches]
        tokenized, swapped, _ = np.split(
            masked_patches,
            [int(.8 * len(masked_patches)),
             int(.9 * len(masked_patches))])
        optimizer.zero_grad()
        output = model(data, unmasked, tokenized, swapped)
        loss = loss_fn(output, data, masked_patches)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.img_size, config.img_size))
    ])
    dataset = datasets.ImageFolder('data', transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config.batch_size)

    model = VisionTransformer(config.hidden_size, config.patch_size,
                              config.model_dim, config.num_heads,
                              config.num_layers, config.img_size).to("cpu")
    optimizer = optim.Adadelta(model.parameters())
    scheduler = StepLR(optimizer, step_size=1)
    perceptual_loss = PerceptualLoss(config.patch_size, config.model_dim,
                                     config.img_size)

    num_patches = (config.img_size // config.patch_size)**2

    for epoch in range(1, config.epochs + 1):
        _train_step(model, perceptual_loss, "cpu", train_loader, optimizer,
                    num_patches, epoch)
        scheduler.step()