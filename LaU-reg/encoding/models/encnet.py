###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import encoding

from .base import BaseNet
from .fcn import FCNHead

from .util import DiffBiUpsampling
from .util import DeterminedBiUpsampling

__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_pcontext',
           'get_encnet_resnet101_pcontext', 'get_encnet_resnet50_ade',
           'get_encnet_resnet101_ade']

class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, offset=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead([512, 1024, 2048], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'],
                            lateral=kwargs['lateral'], norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        self.offset = offset

        if offset:
            self.diffUp0 = DiffBiUpsampling(kwargs['up_factor'], kwargs['category'], kwargs['input_channel'], kwargs['bottleneck_channel'], kwargs['batch_size'], kwargs['downsampled_input_size'], kwargs['downsampled_input_size']) # k, category, input channel, bottleneck channel, batch_size, input_height, input_width
            self.detUp0 = DeterminedBiUpsampling(kwargs['up_factor'], kwargs['category'], kwargs['batch_size'], kwargs['downsampled_input_size'], kwargs['downsampled_input_size']) # k, category, batch_size, input_height, input_width
        
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x, offset_branch = self.head(*features)
        x = list(x)
        
        # LaU : Detach x[0]
        if self.offset:
            if self.training:
                y0, y1, y2, y3, y4, grid = self.detUp0(x[0])
                x.append(y0.detach())
                x.append(grid)
                x.append(y1.detach())
                x.append(y2.detach())
                x.append(y3.detach())
                x.append(y4.detach())

            x[0], pre_offsets = self.diffUp0(x[0], offset_branch)
            if not self.training:
                x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)
        else:
            x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            x.append(auxout)

        if self.training and self.offset:
            x.append(pre_offsets)

        return tuple(x)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            encoding.nn.BatchNorm1d(ncodes),
            nn.ReLU(inplace=True),
            encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], 512, 1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True)) if jpu else \
                     nn.Sequential(nn.Conv2d(in_channels[-1], 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[0], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        offset_branch = outs[0]
        outs[0] = self.conv6(outs[0])
        return tuple(outs), offset_branch


def get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='/mnt/xfs1/home/hexiangyu/tools/encoding/models', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
    }
    # infer number of classes
    from ..datasets import datasets
    model = EncNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('encnet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_encnet_resnet50_pcontext(pretrained=False, root='/mnt/xfs1/home/hexiangyu/tools/encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_encnet('pcontext', 'resnet50', pretrained, root=root, aux=True, 
                      base_size=520, crop_size=480, **kwargs)

def get_encnet_resnet101_pcontext(pretrained=False, root='/mnt/xfs1/home/hexiangyu/tools/encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_encnet('pcontext', 'resnet101', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, lateral=True, **kwargs)

def get_encnet_resnet50_ade(pretrained=False, root='/mnt/xfs1/home/hexiangyu/tools/encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnet('ade20k', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_encnet_resnet101_ade(pretrained=False, root='/mnt/xfs1/home/hexiangyu/tools/encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnet('ade20k', 'resnet101', pretrained, root=root, aux=True,
                      base_size=640, crop_size=576, lateral=True, **kwargs)

def get_encnet_resnet152_ade(pretrained=False, root='/mnt/xfs1/home/hexiangyu/tools/encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnet('ade20k', 'resnet152', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)
