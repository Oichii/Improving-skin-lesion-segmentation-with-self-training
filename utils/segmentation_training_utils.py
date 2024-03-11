from utils.dice_loss import dice_loss, calculate_iou
from utils.training_utils import AverageMeter
import time
import torch
from torchmetrics import JaccardIndex


def train(train_loader, model, criterion, schediuler, optimizer, epoch, cpu=False):
    """
    Runs one train epoch
    :param schediuler:
    :param cpu: use cuda or cpu
    :param train_loader: loader with train images
    :param model: model to train
    :param criterion: loss function
    :param optimizer: used optimizer object
    :param epoch: current epoch number
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    iou = AverageMeter()

    iou_calc = JaccardIndex(num_classes=2).to('cuda')

    # switch to train mode
    model.train()
    end = time.time()

    for i, (net_input, target, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        if not cpu:
            net_input = net_input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

        # compute output
        output = model(net_input)
        loss = criterion(output, target)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), net_input.size(0))
        iou.update(iou_calc(output, target.type(torch.int)).item(), net_input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 30 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(top1.avg, losses.avg, iou.avg)

    schediuler.step()


def validate(val_loader, model, criterion, cpu=False):
    """
    Runs models validation
    :param cpu: use cuda or cpu
    :param val_loader: loader with validation images
    :param model: model to validate
    :param criterion: loss function
    :return: accuracy and loss from validation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    iou = AverageMeter()
    iou_calc = JaccardIndex(num_classes=2).to('cuda')
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (net_input, target, name) in enumerate(val_loader):
        if not cpu:
            net_input = net_input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(net_input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), net_input.size(0))

        iou.update(iou_calc(output, target.type(torch.int)).item(), net_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'iou {top1.val:.3f} ({top1.avg:.3f})')

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return iou.avg, losses.avg
