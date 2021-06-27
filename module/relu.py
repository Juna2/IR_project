import torch
from torch.nn import functional as F


class ReLU(torch.nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def forward(self, input):
        self.activations = F.relu(input, inplace=self.inplace)
        return self.activations

    def lrp(self, R, lrp_mode="simple"):
        return R
