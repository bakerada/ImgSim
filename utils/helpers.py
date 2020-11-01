import json
import torch
import os


def read_config(path):
    with open(path,'r') as f:
        config = json.load(f)
    f.close()
    return config

def setup_experiment(path, id, folders=['checkpoints','logs']):
    path.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in folders:
        p = path / id / f
        p.mkdir(parents=True, exist_ok=True)
        paths.append(p)
    return paths


def save_checkpoint(state, is_best, save_dir ,filename='latest.pt'):
    torch.save(state, os.path.join(save_dir,filename))
    if is_best:
        torch.save(state, os.path.join(save_dir,'best.pt'))

def to_numpy(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()

def initialize_model(model,checkpoint_path,optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model,optimizer,checkpoint['epoch'],checkpoint['best']

def log_info(logger, info, step, dtype='scalar'):
    mapper = {
        'scalar': logger.scalar_summary
    }

    for tag, value, in info.items():
        mapper[dtype](tag, value, step)


def calculate_accuracy(predictions,labels):
    choice = predictions.max(1)[1]
    correct = choice.eq(labels).sum()
    return correct * 100 / predictions.size(0)

