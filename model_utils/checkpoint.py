import torch


def save_ckp(epoch, model, optimizer, scheduler):
    ckpt_path = './checkpoints/checkpoint.pt'
    print("=> Saving model to", ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'model_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)
    print("=> Model sucessfuly saved!")


def load_ckp(ckpt_path, model, optimizer, scheduler):
    print("=> Loading model from", ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    print("=> Model loaded sucesfully!")
    return model, optimizer, scheduler, checkpoint['epoch']