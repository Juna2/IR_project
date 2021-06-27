import torch
import numpy as np

def mask_LRP_seg(interp, seg, loss_filter, args):
    if interp.shape != seg.shape:
        raise Exception('The shape of interp and seg are different!! | interp.shape : {} / seg.shape : {}'.format(interp.shape, seg.shape))
    new_seg = seg * loss_filter.reshape(loss_filter.shape[0], 1, 1, 1).detach() \
                + interp * (1-loss_filter.reshape(loss_filter.shape[0], 1, 1, 1))
    loss = torch.nn.functional.mse_loss(interp, new_seg)
    return loss
    