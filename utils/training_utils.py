"""
Training utilities functions
"""

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

best_prec1 = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def logger(acc, loss, phase='train', history_accuracy=None, history_loss=None, csv_file=None):
    """
    accuracy and loss logger
    :param acc: current accuracy value
    :param loss: current loss value
    :param phase: train or validation phase
    :param history_accuracy:
    :param history_loss:
    :param csv_file: pd dataframe to save accuracy and loss
    """
    if history_accuracy is not None:
        history_accuracy.append(acc)
    if history_loss is not None:
        history_loss.append(loss)
    if csv_file is not None:
        csv_file = csv_file.append({'loss': loss, 'accuracy': acc}, ignore_index=True)
        csv_file.to_csv(phase+'.csv', mode='a', index=False, header=False)


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the trained model
    :param state: state dict
    :param filename: filename to save
    """
    torch.save(state, filename)


def adjust_learning_rate(optim, e, n=30, learning_rate=0.0001):
    """
    Sets the learning rate to the initial LR decayed by 2 every 30 epochs
    :param learning_rate: base learning rate
    :param optim: optimizer object
    :param e: current epoch
    :param n: number of epochs after which update LR
    """

    learning_rate = learning_rate * (0.5 ** (e // n))
    print("current lr: ", learning_rate)
    for param_group in optim.param_groups:
        param_group['lr'] = learning_rate
