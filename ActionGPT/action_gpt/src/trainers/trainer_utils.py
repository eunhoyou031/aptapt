import torch.nn.functional as F
import math

def cross_entropy(pred, target, reduction):
    # print(pred.shape, target.shape)
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction=reduction)
    loss = loss.reshape(target.shape)
    return loss

def masked_loss(pred, target, mask, skip_frame=0, loss_func=F.mse_loss):
    if skip_frame == 0:
        new_pred = pred
    else:
        new_pred = pred[:, :-skip_frame]
    new_target = target[:, skip_frame:]
    new_mask = mask[:, skip_frame:]
    data_shape, mask_shape = new_target.shape, new_mask.shape
    loss = loss_func(new_pred, new_target, reduction='none')
    for _ in range(len(data_shape) - len(mask_shape)):
        new_mask = new_mask.unsqueeze(-1)
    loss = (loss*new_mask).sum() / new_mask.sum() / math.prod(data_shape[len(mask_shape):])
    return loss