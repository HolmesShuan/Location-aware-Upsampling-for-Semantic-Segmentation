#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from modules.location_aware_upsampling import LaU, _LaU, LdU, LdU_MultiOutput

torch.manual_seed(13)

def check_gradient_lau():

    input = torch.rand(2, 16, 5, 5, requires_grad=True).double().cuda()
    offset_x = torch.rand(2, 1, 15, 10, requires_grad=True).double().cuda()
    offset_y = torch.rand(2, 1, 15, 10, requires_grad=True).double().cuda()
    offset_x = offset_x.repeat(1, 16, 1, 1)
    offset_y = offset_y.repeat(1, 16, 1, 1)

    print('check_gradient_lau: ',
          gradcheck(_LaU, (input, offset_x, offset_y, 3, 2),
                    eps=1e-5, atol=1e-8, rtol=1e-8, raise_exception=True))

def example_lau():
    k = 2
    input = torch.ones(2, 2, 2, 2, requires_grad=True).cuda()
    offset_x = torch.zeros(2, 1, 4, 4, requires_grad=True).cuda()
    offset_y = torch.zeros(2, 1, 4, 4, requires_grad=True).cuda() + 0.5
    offset_x = offset_x.repeat(1, 2, 1, 1)
    offset_y = offset_y.repeat(1, 2, 1, 1)

    input[:,:,0,0] = 0.
    input[:,:,1,1] = 0.

    lau = LaU(2, 2).cuda()
    output = lau(input, offset_x, offset_y)
    print(input)
    print(output)

def example_ldu():
    k = 2
    input = torch.ones(2, 2, 2, 2, requires_grad=True).cuda()

    input[:,:,0,0] = 0.
    input[:,:,1,1] = 0.

    ldu = LdU(2, 2).cuda()
    output = ldu(input)
    print(input)
    print(output)

def example_ldu_multi_output():
    k = 2
    input = torch.ones(2, 2, 2, 2, requires_grad=True).cuda()

    input[:,:,0,0] = 0.
    input[:,:,1,1] = 0.

    ldu_multioutput = LdU_MultiOutput(2, 2).cuda()
    output, output_lt, output_lb, output_rt, output_rb = ldu_multioutput(input)
    print(input)
    print(output)
    print(output_lt)
    print(output_lb)
    print(output_rt)
    print(output_rb)

if __name__ == '__main__':

    example_lau()
    example_ldu()
    example_ldu_multi_output()
    check_gradient_lau()
