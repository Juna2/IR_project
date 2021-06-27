import torch
import torch.nn as nn

from module.sequential import Sequential
from module.linear import Linear
from module.relu import ReLU
from module.convolution import Conv2d
from module.pool import MaxPool2d
from module.flatten import Flatten
from module.dropout import Dropout2d

from torch.hub import load_state_dict_from_url
from utils import register



__all__ = [
    "VGG",
    "vgg16",
    "vgg19",
]


model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.flatten = Flatten()
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(),
            Dropout2d(),
            Linear(4096, 4096),
            ReLU(),
            Dropout2d(),
            Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def lrp(self, R, args, lrp_mode="simple"):
        R = self.classifier.lrp(R, args, lrp_mode=lrp_mode)
        R = self.flatten.lrp(R, args, lrp_mode=lrp_mode)
        R = self.features.lrp(R, args, lrp_mode=lrp_mode)
        
        return R

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU()]
            else:
                layers += [conv2d, ReLU()]
            in_channels = v
    return Sequential(*layers)


cfgs = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict['classifier.6.weight'] = model.state_dict()['classifier.6.weight']
        state_dict['classifier.6.bias'] = model.state_dict()['classifier.6.bias']
        model.load_state_dict(state_dict)
    return model

@register.setmodelname("vgg16")
def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)

@register.setmodelname("vgg19")
def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)
