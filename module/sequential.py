import torch
import torch.nn.functional as F

from module.pool import MaxPool2d
from module.convolution import Conv2d
from module.relu import ReLU


class Sequential(torch.nn.Sequential):
    def lrp(self, R, args, lrp_mode="simple"):
        for key, module in reversed(list(self._modules.items())):
            if (isinstance(module, Conv2d) or isinstance(module, MaxPool2d)) and key == args.lrp_target_layer:
                break
            R = module.lrp(R, lrp_mode=lrp_mode)
            
        return R