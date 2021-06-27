# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class Conv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
        )
        self.alpha = 1
        self.beta = 0

    def forward(self, input):
        self.input_tensor = input
        output = self._conv_forward(input=input, weight=self.weight, bias=self.bias)
        self.output_shape = output.shape
        return output

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def lrp(self, R, lrp_mode="simple"):
        if lrp_mode == "simple":
            return self._simple_lrp(R)
        elif lrp_mode == "composite":
            return self._composite_lrp(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _simple_lrp(self, R):
        def f(x, w, b):
            Zs = self._conv_forward(x, w, b)
            stabilizer = 1e-2 * (torch.where(torch.ge(Zs, 0), torch.ones_like(Zs), torch.ones_like(Zs) * -1))

            RdivZs = R / (Zs + stabilizer)
            Rx = x * torch.nn.grad.conv2d_input(x.shape, w, grad_output=RdivZs, padding=self.padding)
            return Rx

        return f(self.input_tensor, self.weight, self.bias)

    def _composite_lrp(self, R):
        weight_p = F.relu(self.weight)
        weight_n = self.weight - weight_p
        input_p = F.relu(self.input_tensor)
        input_n = self.input_tensor - input_p
        bias_p = F.relu(self.bias)
        bias_n = self.bias - bias_p

        def f(x1, x2, w1, w2, b):
            Z1 = self._conv_forward(x1, w1, b)
            Z2 = self._conv_forward(x2, w2, b)
            Zs = Z1 + Z2

            stabilizer = 1e-2 * (torch.where(torch.ge(Zs, 0), torch.ones_like(Zs), torch.ones_like(Zs) * -1))
            RdivZs = R / (Zs + stabilizer)

            tmp1 = x1 * torch.nn.grad.conv2d_input(
                x1.shape, w1, grad_output=RdivZs, stride=self.stride, padding=self.padding
            )

            tmp2 = x2 * torch.nn.grad.conv2d_input(
                x2.shape, w2, grad_output=RdivZs, stride=self.stride, padding=self.padding
            )
            return tmp1 + tmp2

        R_alpha = f(input_p, input_n, weight_p, weight_n, bias_p)

        if self.beta != 0:
            R_beta = f(input_p, input_n, weight_n, weight_p, bias_n)
            Rx = (1 - self.beta) * R_alpha + self.beta * R_beta
        else:
            Rx = R_alpha

        return Rx
