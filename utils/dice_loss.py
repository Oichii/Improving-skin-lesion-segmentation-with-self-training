import torch
from torch import Tensor
import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    """
    require sigmoid at the end of network
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        dscs = torch.zeros(y_pred.shape[1])

        for i in range(y_pred.shape[1]):
          y_pred_ch = y_pred[:, i].contiguous().view(-1)
          y_true_ch = y_true[:, i].contiguous().view(-1)
          intersection = (y_pred_ch * y_true_ch).sum()
          dscs[i] = (2. * intersection + self.smooth) / (
              y_pred_ch.sum() + y_true_ch.sum() + self.smooth
          )
        return 1. - torch.mean(dscs)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def calculate_iou(predicted, target):
    # pass
    with torch.no_grad():
        overlap = predicted.logical_and(target)
        union = predicted.logical_or(target)
        return overlap / union
