import torch
from torch.nn import functional as F


class MaxPool2d(torch.nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        self.input_tensor = input

        self.activations, self.indices = F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, True
        )
        return self.activations

    def simple_grad(self, grad_output):
        grad_input = F.max_unpool2d(
            grad_output.reshape(self.activations.shape),
            self.indices,
            self.kernel_size,
            self.stride,
            self.padding,
            self.input_tensor.shape,
        )
        return grad_input

    def lrp(self, R, lrp_mode="simple"):
        R = self.simple_grad(R)
        return R


# class AvgPool2d(torch.nn.AvgPool2d):
#     def __init__(
#         self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None,
#     ):
#         super().__init__(
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             ceil_mode=ceil_mode,
#             count_include_pad=count_include_pad,
#             divisor_override=divisor_override,
#         )

#         self.beta = 0

#     def forward(self, input):
#         self.input_tensor = input
#         self.output = F.avg_pool2d(
#             input,
#             self.kernel_size,
#             self.stride,
#             self.padding,
#             self.ceil_mode,
#             self.count_include_pad,
#             self.divisor_override,
#         )
#         return self.output

#     def simple_grad(self, grad_output):
#         grad_input = F.interpolate(grad_output, scale_factor=self.kernel_size, mode="nearest")
#         grad_input = grad_input / (self.kernel_size ** 2)
#         return grad_input

#     def grad_cam(self, grad_output):
#         grad_input = self.simple_grad(grad_output)
#         return grad_input, self.input_tensor

#     def lrp(self, R, lrp_mode="simple"):
#         if lrp_mode == "simple":
#             return self._simple_lrp(R)
#         elif lrp_mode == "composite":
#             return self._composite_lrp(R)
#         raise NameError(f"{lrp_mode} is not a valid lrp name")

#     def _simple_lrp(self, R):
#         RdivZ = R / (self.output + 1e-8) / (self.kernel_size ** 2)
#         RdivZ_interpol = F.interpolate(RdivZ, scale_factor=self.kernel_size, mode="nearest")

#         return self.input_tensor * RdivZ_interpol

#     def _composite_lrp(self, R):

#         x_p = torch.where(self.input_tensor < 0, torch.zeros(1), self.input_tensor)
#         x_n = self.input_tensor - x_p

#         output_p = F.avg_pool2d(
#             x_p, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad
#         )
#         output_n = F.avg_pool2d(
#             x_n, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad
#         )

#         RdivZp = R / (output_p + 1e-8) / (self.kernel_size ** 2)
#         RdivZn = R / (output_n + 1e-8) / (self.kernel_size ** 2)

#         RdivZp_interpol = F.interpolate(RdivZp, scale_factor=self.kernel_size, mode="nearest")
#         RdivZn_interpol = F.interpolate(RdivZn, scale_factor=self.kernel_size, mode="nearest")

#         return (1 - self.beta) * x_p * RdivZp_interpol + self.beta * x_n * RdivZn_interpol


# class AdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
#     def __init__(self, output_size):
#         super().__init__(output_size=output_size)
#         self.beta = 0

#     def forward(self, input):
#         self.input_tensor = input
#         self.output = F.adaptive_avg_pool2d(input, self.output_size)
#         return self.output

#     def simple_grad(self, grad_output):
#         scale = self.input_tensor.shape[2] / grad_output.shape[2]
#         grad_output = F.interpolate(grad_output, scale_factor=scale, mode="nearest")
#         grad_input = grad_output / (scale * scale)
#         return grad_input

#     def grad_cam(self, grad_output):
#         grad_input = self.simple_grad(grad_output)
#         return grad_input, self.input_tensor

#     def lrp(self, R, lrp_mode="simple"):
#         if lrp_mode == "simple":
#             return self._simple_lrp(R)
#         elif lrp_mode == "composite":
#             return self._composite_lrp(R)
#         raise NameError(f"{lrp_mode} is not a valid lrp name")

#     def _simple_lrp(self, R):
#         lrp_input = (
#             self.input_tensor * R / ((self.input_tensor.shape[2] * self.input_tensor.shape[3]) * self.output + 1e-8)
#         )

#         return lrp_input

#     def _composite_lrp(self, R):
#         x_p = torch.where(self.input_tensor < 0, torch.zeros(1), self.input_tensor)
#         x_n = self.input_tensor - x_p

#         Rx_p = x_p * R / (x_p.sum(dim=(2, 3), keepdim=True) + 1e-8)
#         Rx_n = x_n * R / (x_n.sum(dim=(2, 3), keepdim=True) + 1e-8)
#         return (1 - self.beta) * Rx_p + (self.beta * Rx_n)

