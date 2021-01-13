import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from model_utils.vision_transformer import VisionTransformer
from model_utils.perceptual_loss import PerceptualLoss
from model_utils.average_meter import AverageMeter
from model_utils.plot_gradients import plot_grad_flow
from model_utils.checkpoint import save_ckp, load_ckp


def _transform_targets(targets, masked_patches, patch_size, num_patch_axis):
    masked_patches_dims = []
    for masked_patch in masked_patches:
        height_offset = patch_size * ((masked_patch) // num_patch_axis)
        width_offset = patch_size * ((masked_patch) % num_patch_axis)
        masked_patches_dims.append([height_offset, width_offset])

    target_patches = []
    for target in range(targets.shape[0]):
        for masked_patch in masked_patches_dims:
            height_offset, width_offset = masked_patch
            extracted_patch = targets[target, :,
                                      height_offset:height_offset + patch_size,
                                      width_offset:width_offset + patch_size]
            target_patches.append(extracted_patch)

    target_patches_tensor = torch.stack(target_patches)
    target_patches_tensor = (target_patches_tensor // 0.34).long()
    encoded_targets = F.one_hot(target_patches_tensor, 3)
    n, c, w, h, e = encoded_targets.shape
    encoded_targets = torch.reshape(encoded_targets, [n, w, h, c * e])
    mean_targets = torch.mean(encoded_targets.float(),
                              dim=[1, 2],
                              keepdim=True).view(n, c * e)
    return mean_targets


def _transform_outputs(outputs, masked_patches):
    output_dim = outputs.shape[-1]

    masked_patches_shifted = [
        masked_patch + 1 for masked_patch in masked_patches
    ]
    outputs = outputs[:, masked_patches_shifted, :]
    outputs = outputs.view(-1, output_dim)
    return outputs


def _train_step(model,
                device,
                train_loader,
                train_writer,
                optimizer,
                num_patches,
                patch_size,
                num_patch_axis,
                epoch,
                debug=False):
    losses = AverageMeter('Loss', ':.4e')

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
        transformed_outputs = _transform_outputs(output, masked_patches)
        transformed_targets = _transform_targets(data, masked_patches,
                                                 patch_size, num_patch_axis)
        loss = F.mse_loss(transformed_outputs, transformed_targets)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        if debug:
            plot_grad_flow(model.named_parameters())
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_writer.add_scalar("loss", loss.item(),
                                    ((epoch - 1) * len(train_loader.dataset)) +
                                    (batch_idx * len(data)))


def _single_step(model,
                 loss_fn,
                 device,
                 train_loader,
                 train_writer,
                 optimizer,
                 num_patches,
                 epoch,
                 debug=False):
    losses = AverageMeter('Loss', ':.4e')

    patches = list(range(num_patches))
    model.train()
    data, target = next(iter(train_loader))
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
    losses.update(loss.item(), data.size(0))
    loss.backward()
    if debug:
        plot_grad_flow(model.named_parameters())
    optimizer.step()


def train(config, resume_training=False, debug=False):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.img_size, config.img_size))
    ])
    dataset = datasets.ImageFolder(config.data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config.batch_size)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = SummaryWriter("./runs/" + log_time + "/train")

    model = VisionTransformer(config.hidden_size, config.patch_size,
                              config.model_dim, config.output_dim,
                              config.num_heads, config.num_layers,
                              config.img_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1)

    num_patch_axis = config.img_size // config.patch_size
    num_patches = num_patch_axis**2

    start_epoch = 1
    if resume_training:
        model, optimizer, scheduler, start_epoch = load_ckp(
            config.ckpt_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, config.epochs + 1):
        _train_step(model, device, train_loader, train_writer, optimizer,
                    num_patches, config.patch_size, num_patch_axis, epoch,
                    debug)
        scheduler.step()
        save_ckp(epoch, model, optimizer, scheduler)