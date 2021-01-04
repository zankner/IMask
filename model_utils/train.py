import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from vision_transformer import VisionTransformer
from perceptual_loss import PerceptualLoss


def _train_step(model, loss_fn, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        masked_patch = np.random.randint(0, 14)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data, masked_patch)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224))])
    dataset = datasets.ImageFolder('../data', transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    model = VisionTransformer().to("cpu")
    optimizer = optim.Adadelta(model.parameters())
    scheduler = StepLR(optimizer, step_size=1)
    perceptual_loss = PerceptualLoss()

    for epoch in range(1, 2):
        _train_step(model, perceptual_loss, "cpu", train_loader, optimizer,
                    epoch)
        scheduler.step()


train()