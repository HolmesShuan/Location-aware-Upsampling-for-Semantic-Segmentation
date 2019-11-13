#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import LAU

class LAUFunction(Function):
    @staticmethod
    def forward(ctx, input, offset_x, offset_y, k_h, k_w):
        ctx.k_h = k_h
        ctx.k_w = k_w
        output = LAU.location_aware_upsampling_forward(input, 
                                        offset_x, 
                                        offset_y,
                                        ctx.k_h, 
                                        ctx.k_w)
        ctx.save_for_backward(input, offset_x, offset_y)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset_x, offset_y = ctx.saved_tensors
        grad_input, grad_offset_x, grad_offset_y = \
            LAU.location_aware_upsampling_backward(input,
                                     offset_x,
                                     offset_y,
                                     grad_output)

        return grad_input, grad_offset_x, grad_offset_y, None, None

class LDUFunction(Function):
    @staticmethod
    def forward(ctx, input, k_h, k_w):
        ctx.k_h = k_h
        ctx.k_w = k_w
        output = LAU.location_determined_upsampling_forward(input, 
                                        ctx.k_h, 
                                        ctx.k_w)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = LAU.location_determined_upsampling_backward(input,
                                     grad_output)

        return grad_input, None, None

class LDUMultiOutputFunction(Function):
    @staticmethod
    def forward(ctx, input, k_h, k_w):
        ctx.k_h = k_h
        ctx.k_w = k_w
        output, output_lt, output_lb, output_rt, output_rb = LAU.location_determined_upsampling_multi_output_forward(input, 
                                        ctx.k_h,
                                        ctx.k_w)
        ctx.save_for_backward(input)
        return output, output_lt, output_lb, output_rt, output_rb

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = LAU.location_determined_upsampling_multi_output_backward(input,
                                     grad_output)

        return grad_input, None, None