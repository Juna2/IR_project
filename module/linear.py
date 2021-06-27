import torch
from torch.nn import functional as F


class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input):
        self.input_tensor = input
        return F.linear(self.input_tensor, self.weight, self.bias)

    def lrp(self, R, lrp_mode="simple"):
        if lrp_mode == "simple":
            return self._simple_lrp(R)
        elif lrp_mode == "composite":
            return self._composite_lrp(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _simple_lrp(self, R):
        ######################## Your answer should be in here #################
        
        Zs = F.linear(self.input_tensor, self.weight, self.bias)
        stabilizer = 1e-2 * (torch.where(torch.ge(Zs, 0), torch.ones_like(Zs), torch.ones_like(Zs) * -1))
        Zs += stabilizer

        RdivZs = R / Zs
        Rx = RdivZs.mm(self.weight) * self.input_tensor
        ######################## Your answer should be in here #################
        return Rx

    def _composite_lrp(self, R):
        return self._simple_lrp(R)
