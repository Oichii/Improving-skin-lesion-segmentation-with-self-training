"""
Main training script
"""
import pandas as pd
import os

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from utils.utils import set_seed

from utils.segmentation_training_utils import validate, train
from datasets.skinSegmentationDatasetOptical import SkinSegmentationDataset, get_transforms_segmentation
from models.select_model import select_model
from datetime import datetime

import json
import segmentation_models_pytorch as smp


def main(config=None, s=1234):

    with open('config_segmentation.json') as config_file:
        paths = json.load(config_file)
    ratio = paths['csvPath'].split('_')[2]
    best_prec1 = 0

    save_dir = paths['savePath']
    cpu = False
    resume = paths['resumePath']

    start_epoch = 0

    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    cudnn.benchmark = True
    set_seed(s)

    # define loss function (criterion)
    if config['loss'] == 'bce':
        weights = torch.tensor(config['weights']).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    else:
        criterion = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)

    criterion.to(device)

    transforms_train, transforms_val = get_transforms_segmentation(config["img_size"], transform=config['transform'])
    config['transforms'] = transforms_train.__str__()

    # Check if the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(paths['csvPath'])

    folds = [1]
    backbone = config['backbone']
    for fold in folds:
        # df_train = df
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]
        print("train samples:", len(df_train), "validation sample:", len(df_valid))

        # prepare model
        model = select_model(config['net_name'], width=1, backbone=backbone)
        print(model)
        model.to(device)

        # prepare optimizer
        if config['optim'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), config['lr'],
                                         weight_decay=config["weight_decay"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), config["lr"],
                                        momentum=config["momentum"],
                                        weight_decay=config["weight_decay"])

        scheduler_cosine = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1)

        # optionally resume from a checkpoint
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> loaded checkpoint (epoch {})"
                      .format(checkpoint['epoch']))
                print(checkpoint.keys())
            else:
                print("=> no checkpoint found at '{}'".format(resume))

        train_dataset = SkinSegmentationDataset(df_train,
                                                img_path=paths['imagesPath'],
                                                mask_path=paths['masksPath'],
                                                transforms=transforms_train,
                                                )
        test_dataset = SkinSegmentationDataset(df_valid,
                                               img_path=paths['imagesPath'],
                                               mask_path=paths['masksPath'],
                                               transforms=transforms_val,
                                               )
        batch_size = config["batch_size"]
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=config['num_workers'],
                                                   pin_memory=config['pin_memory'],
                                                   drop_last=True
                                                   )
        val_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=config['num_workers'],
                                                 pin_memory=config['pin_memory']
                                                 )

        # visualisation of sample data
        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        for epoch in range(start_epoch, config["epochs"]):
            train(train_loader, model, criterion, scheduler_cosine, optimizer, epoch, cpu)
            # evaluate on validation set
            prec1, _ = validate(val_loader, model, criterion, cpu)

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)
            model_save_path = os.path.join(save_dir,
                                           f'student_new_'
                                           f'{dt_string}_{config["net_name"]}_{backbone}'
                                           f'_optim={optimizer.__class__.__name__}'
                                           f'_loss={config["loss"]}'
                                           f'_e={epoch}'
                                           f'_lr={config["lr"]}'
                                           f'_bs={config["batch_size"]}'
                                           f'_color={config["color_space"]}'
                                           f'_w={config["weights"]}'
                                           f'_{config["momentum"]}'
                                           f'f={fold}_optical_s_r={ratio}_{s}.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, model_save_path)


if __name__ == '__main__':
    cfg = {
        "batch_size": 10,
        "width": 1,
        'img_size': 256,
        'optim': 'sgd',
        'lr': 0.006,
        'momentum': 0.81,
        'weight_decay': 0.02,
        'epochs': 65,
        'net_name': 'unet',
        'num_workers': 0,
        'pin_memory': True,
        "weights": [1.25],
        'loss': 'dice',
        "backbone": "resnet18",
        'transform': "optical"
    }
    seeds = [2137]
    for seed in seeds:
        main(cfg, seed)
