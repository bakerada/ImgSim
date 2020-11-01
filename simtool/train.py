from utils.helpers import *
from dataloader import create_dataloader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.logger import Logger
from pathlib import Path
from model import RockClassifier
import time
import os

optimizer_map = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}
lrschedule_map = {
    "step": optim.lr_scheduler.StepLR,
    "multistep": optim.lr_scheduler.MultiStepLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau
}

def train(loader, model, optimizer, epoch, logger, log_interval=1000):
    model.train()
    use_cuda = next(model.parameters()).is_cuda
    start = time.time()
    steps = len(loader)
    for batch_idx,data in enumerate(loader):
        optimizer.zero_grad()
        imgs,labels = data
        imgs, labels = imgs.float(), labels.long() -1
        if use_cuda:
            imgs,labels = imgs.cuda(), labels.cuda()

        output,_ = model(imgs)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(output.detach(),labels).item()
        if batch_idx % log_interval== 0:
            info = {
                     'loss': loss.item(),
                     'Train Accuracy': accuracy
                    }
            step = ((steps * epoch) + batch_idx) * loader.batch_size
            print(step)
            log_info(logger, info, step)
    return  len(loader) / (time.time() - start)

def test(loader, model, optimizer, epoch, logger):
    model.eval()
    use_cuda = next(model.parameters()).is_cuda
    start = time.time()
    total_loss = 0.
    total_accuaracy = 0.
    for batch_idx,data in enumerate(loader):
        imgs,labels = data
        imgs, labels = imgs.float(), labels.long() -1
        if use_cuda:
            imgs,labels = imgs.cuda(), labels.cuda()

        output,_ = model(imgs)
        total_loss += F.nll_loss(output, labels).detach().item()
        total_accuaracy += calculate_accuracy(output.detach(),labels).item()
    info = {
        'Test Accuracy': (total_accuaracy / len(loader)) ,
        'Test Loss': total_loss / len(loader)
    }

    log_info(logger, info, epoch)
    return  total_accuaracy / len(loader)


def main():
    config = read_config('config/train.json')
    train_loader = create_dataloader(config['traindir'],config['data'])
    test_loader = create_dataloader(config['testdir'],config['data'])
    basedir = Path(config['basedir'])
    checkpoint_path, log_path = setup_experiment(basedir,config['id'])
    logger = Logger(str(log_path))
    torch.manual_seed(config['seed'])
    model = RockClassifier(**config['model'])
    if (torch.cuda.is_available()) and (len(config['gpus']) > 0):
        model = nn.DataParallel(model,device_ids = config['gpus'])

    optimizer = optimizer_map.get(config['optimizer'],'adam')(model.parameters(),**config['optimizer_args'])
    scheduler = lrschedule_map.get(config['lr_schedule'],'step')(optimizer,**config['lr_schedule_args'])

    epoch = 0
    best_score = -np.inf
    if config.get('checkpoint_path',False):
        print("loading model from {}".format(config['checkpoint_path']))
        model,optimizer,epoch,best_score = initialize_model(model,config['checkpoint_path'],optimizer=optimizer)
    print("loading from epoch: {}".format(epoch))

    for e in range(epoch,config['epochs']):
        if config.get('lr_schedule','step') == 'plateau':
            scheduler.step(best_score * -1)
        else:
            scheduler.step()
        rate = train(train_loader, model, optimizer, e, logger,config['log_interval'])
        info = {
        'Rate': rate
        }

        log_info(logger, info, e)

        if e % config['test_interval']==0:
            score = test(test_loader,model,config,e,logger)
            is_best = score > best_score
            best_score = max(best_score,score)

            save_checkpoint({
							'epoch': epoch ,
							'state_dict': model.state_dict(),
							'best': best_score,
							'optimizer' : optimizer.state_dict(),
                            'epoch':e
						}, is_best, str(checkpoint_path))


if __name__ == '__main__':
    main()
