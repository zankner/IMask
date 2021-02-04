import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from model_utils.vision_transformer import VisionTransformer
from model_utils.classifier import Classifier
from model_utils.average_meter import AverageMeter
from model_utils.plot_gradients import plot_grad_flow
from model_utils.checkpoint import save_ckp, load_ckp, load_pretrained_ckpt


def _accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def _train_step(model,
                device,
                train_loader,
                train_writer,
                optimizer,
                epoch,
                debug=False):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        losses.update(loss.item(), data.size(0))
        acc1, acc5 = _accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        loss.backward()
        if debug:
            plot_grad_flow(model.named_parameters())
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Top1: {:.6f} Top5: {:.6f}'
                .format(epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(),
                        acc1[0], acc5[0]))
            train_writer.add_scalar("loss", loss.item(),
                                    ((epoch - 1) * len(train_loader.dataset)) +
                                    (batch_idx * len(data)))


def _test_step(model, device, test_loader, test_writer, epoch, debug=False):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            losses.update(loss.item(), data.size(0))
            acc1, acc5 = _accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
    print('=============================================================')
    print('Test Epoch: {} \tLoss: {:.6f} Top1: {:.6f} Top5: {:.6f}'.format(
        epoch, losses.avg, top1.avg, top5.avg))
    print('=============================================================')
    test_writer.add_scalar("loss", losses.avg, epoch)
    test_writer.add_scalar("top1", top1.avg, epoch)
    test_writer.add_scalar("top5", top5.avg, epoch)


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


def fine_tune(config, resume_training=False, debug=False):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.img_size, config.img_size))
    ])
    train_dataset = datasets.CIFAR100(config.data_dir + '/train',
                                      train=True,
                                      transform=transform)
    test_dataset = datasets.CIFAR100(config.data_dir + '/test',
                                     train=False,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.batch_size)
    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = SummaryWriter("./runs/" + log_time + "/train")
    test_writer = SummaryWriter("./runs/" + log_time + "/test")

    vision_transformer = VisionTransformer(config.hidden_size,
                                           config.patch_size, config.model_dim,
                                           config.output_dim, config.num_heads,
                                           config.num_layers, config.img_size)
    vision_transformer.to(device)
    vision_transformer = load_pretrained_ckpt(
        config.vision_transformer_ckpt_path, vision_transformer)
    model = Classifier(vision_transformer, config.model_dim,
                       config.dense_1_dim, config.dense_2_dim,
                       config.num_classes, config.classifier_dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1)

    start_epoch = 1
    if resume_training:
        model, optimizer, scheduler, start_epoch = load_ckp(
            config.ckpt_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, config.epochs + 1):
        _train_step(model, device, train_loader, train_writer, optimizer,
                    epoch, debug)
        _test_step(model, device, test_loader, test_writer, epoch, debug)
        scheduler.step()
        save_ckp(epoch, model, optimizer, scheduler)