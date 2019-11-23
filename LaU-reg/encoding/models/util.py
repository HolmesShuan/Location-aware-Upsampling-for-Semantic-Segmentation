import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lau import *
import encoding

from .LaU import LaU, LdU, LdU_MultiOutput

class DiffBiUpsampling(nn.Module):
    def __init__(self, k, category, offset_branch_input_channels, bottleneck_channel, batch_size, input_height, input_width, **kwargs):
        super(DiffBiUpsampling, self).__init__()

        self.k = k
        self.infer_w = nn.Sequential(
                            nn.Conv2d(offset_branch_input_channels, bottleneck_channel, 1, padding=0, bias=False, **kwargs),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(bottleneck_channel, self.k*self.k, 3, padding=1, bias=False, **kwargs)
                        )
        self.infer_h = nn.Sequential(
                            nn.Conv2d(offset_branch_input_channels, bottleneck_channel, 1, padding=0, bias=False, **kwargs),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(bottleneck_channel, self.k*self.k, 3, padding=1, bias=False, **kwargs)
                        )

        self.pixelshuffle = nn.PixelShuffle(self.k)

        nn.init.xavier_uniform_(self.infer_w[0].weight)
        nn.init.xavier_uniform_(self.infer_h[0].weight)
        nn.init.xavier_uniform_(self.infer_w[2].weight)
        nn.init.xavier_uniform_(self.infer_h[2].weight)

        self.lau = LaU(self.k, self.k).cuda()
    
    def forward(self, x, offset_branch):
        offsets_h = self.infer_h(offset_branch)
        offsets_w = self.infer_w(offset_branch)

        offsets_h = self.pixelshuffle(offsets_h)
        offsets_w = self.pixelshuffle(offsets_w)

        if self.training:
            offsets_return = torch.cat((offsets_h, offsets_w), dim=1) # (b, 2c, H, W)
        else:
            offsets_return = None

        offsets_h = offsets_h.repeat(1, x.size(1), 1, 1)
        offsets_w = offsets_w.repeat(1, x.size(1), 1, 1)

        y_offset = self.lau(x, offsets_h, offsets_w)
        return y_offset, offsets_return

class DeterminedBiUpsampling(nn.Module):
    def __init__(self, k, channels, batch_size, input_height, input_width, **kwargs):
        super(DeterminedBiUpsampling, self).__init__()

        self.filters = channels
        self.k = k
        self.H = int(math.floor(input_height*self.k))
        self.W = int(math.floor(input_width*self.k))

        self._grid_for_loss_param = (batch_size, self.H, self.W)
        self._grid_for_loss = th_generate_grid_for_loss(batch_size, self.H, self.W, torch.cuda.FloatTensor, True)

        self.ldu = LdU_MultiOutput(self.k, self.k).cuda()
    
    def forward(self, x):
        with torch.no_grad():
            y0, y1, y2, y3, y4 = self.ldu(x)

            grid = self._get_grid_for_loss(y0)

        return y0, y1, y2, y3, y4, grid.detach().to(y0.device).div(self.k)

    def _get_grid_for_loss(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(2), x.size(3)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_for_loss_param == (batch_size, input_height, input_width):
            return self._grid_for_loss
        self._grid_for_loss_param = (batch_size, input_height, input_width)
        self._grid_for_loss = th_generate_grid_for_loss(batch_size, input_height, input_width, dtype, cuda)
        return self._grid_for_loss