from torch import nn
import pandas as pd


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
