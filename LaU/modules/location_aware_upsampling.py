#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair

from ..functions.location_aware_upsampling_func import LAUFunction, LDUFunction, LDUMultiOutputFunction

class LaU(nn.Module):
    def __init__(self, k_h, k_w):
        super(LaU, self).__init__()
        self.k_h = k_h
        self.k_w = k_w
        assert type(k_h) == int
        assert type(k_w) == int

    def forward(self, input, offset_x, offset_y):
        assert offset_x.size() == offset_y.size()
        return LAUFunction.apply(input, offset_x, offset_y, self.k_h, self.k_w)

_LaU = LAUFunction.apply

class LdU(nn.Module):
    def __init__(self, k_h, k_w):
        super(LdU, self).__init__()
        self.k_h = k_h
        self.k_w = k_w
        assert type(k_h) == int
        assert type(k_w) == int

    def forward(self, input):
        return LDUFunction.apply(input, self.k_h, self.k_w)

class LdU_MultiOutput(nn.Module):
    def __init__(self, k_h, k_w):
        super(LdU_MultiOutput, self).__init__()
        self.k_h = k_h
        self.k_w = k_w
        assert type(k_h) == int
        assert type(k_w) == int

    def forward(self, input):
        return LDUMultiOutputFunction.apply(input, self.k_h, self.k_w)        