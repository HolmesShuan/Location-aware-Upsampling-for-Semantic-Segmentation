##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss, NLLLoss

from torch.autograd import Variable

torch_ver = torch.__version__[:3]

__all__ = ['SegmentationLosses', 'OffsetLosses', 'PyramidPooling', 'JPU', 'Mean']

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average=True, ignore_index=ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            # pred1, se_pred, pred2, target = tuple(inputs)
            pred1_diffdup, se_pred, pred1_detup, pred2_detup, pred2_diffup, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)
            loss1 = super(SegmentationLosses, self).forward(pred1_diffdup, target)
            loss2 = super(SegmentationLosses, self).forward(pred2_diffup, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class OffsetLosses(Module):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, offset=True, offset_weight=0.3, location_regression_weight=0.3,
                 weight=None, size_average=True, ignore_index=-1):
        super(OffsetLosses, self).__init__()
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.offset = offset
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.offset_weight = offset_weight
        self.location_regression_weight = location_regression_weight
        self.bceloss = BCELoss(weight, size_average)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss(reduction='none', ignore_index=ignore_index)
        self.smoothl1 = nn.SmoothL1Loss(reduction='mean')
        self.crossentropy = nn.CrossEntropyLoss(weight, size_average=size_average, ignore_index=ignore_index)

    def forward(self, *inputs):
        if self.se_loss and self.aux:
            pred1_diffdup, se_pred, pred1_detup, grid, pred1_lt_detup, pred1_lb_detup, pred1_rt_detup, pred1_rb_detup, pred2, offsets, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
            
            with torch.no_grad():
                pred1_detup_logsoftmax  = self.logsoftmax(pred1_detup)
                pred1_lt_detup_logsoftmax = self.logsoftmax(pred1_lt_detup)
                pred1_lb_detup_logsoftmax = self.logsoftmax(pred1_lb_detup)
                pred1_rt_detup_logsoftmax = self.logsoftmax(pred1_rt_detup)
                pred1_rb_detup_logsoftmax = self.logsoftmax(pred1_rb_detup)

                pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss3 = self.nllloss(pred1_lt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss4 = self.nllloss(pred1_lb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss5 = self.nllloss(pred1_rt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss6 = self.nllloss(pred1_rb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)

                coords_lt = grid.floor().float() - grid.float()
                coords_rb = grid.ceil().float() - grid.float()
                coords_lb = torch.cat((coords_rb[:,0,:,:].unsqueeze(dim=1), coords_lt[:,1,:,:].unsqueeze(dim=1)), 1) # coords_lt[..., 0] : row | coords_lt[..., 1] : col
                coords_rt = torch.cat((coords_lt[:,0,:,:].unsqueeze(dim=1), coords_rb[:,1,:,:].unsqueeze(dim=1)), 1)

                gt_offsets = torch.zeros(offsets.shape).to(offsets.device)
                gt_offsets = gt_offsets + offsets
                min_error = pred1_loss1

                error_map = torch.lt(pred1_loss3, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lt-gt_offsets)
                min_error = torch.min(pred1_loss3, min_error)

                error_map = torch.lt(pred1_loss4, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lb-gt_offsets)
                min_error = torch.min(pred1_loss4, min_error)

                error_map = torch.lt(pred1_loss5, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rt-gt_offsets)
                min_error = torch.min(pred1_loss5, min_error)

                error_map = torch.lt(pred1_loss6, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rb-gt_offsets)
                min_error = torch.min(pred1_loss6, min_error)

                error_map_loss1 = torch.gt(pred1_loss1, min_error).float()
                error_map_loss1 = error_map_loss1.mul(self.offset_weight)
                error_map_loss1.add_(1.0)

            pred1_loss1.mul_(error_map_loss1.detach())
            
            offset_loss = self.smoothl1(gt_offsets.detach(), offsets)

            loss1 = torch.mean(pred1_loss1)
            loss2 = self.crossentropy(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            loss4 = offset_loss

            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3 + self.location_regression_weight * loss4 
        elif not self.se_loss:
            pred1_diffdup, pred1_detup, grid, pred1_lt_detup, pred1_lb_detup, pred1_rt_detup, pred1_rb_detup, pred2, offsets, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
            
            with torch.no_grad():
                pred1_detup_logsoftmax  = self.logsoftmax(pred1_detup)
                pred1_lt_detup_logsoftmax = self.logsoftmax(pred1_lt_detup)
                pred1_lb_detup_logsoftmax = self.logsoftmax(pred1_lb_detup)
                pred1_rt_detup_logsoftmax = self.logsoftmax(pred1_rt_detup)
                pred1_rb_detup_logsoftmax = self.logsoftmax(pred1_rb_detup)

                pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss3 = self.nllloss(pred1_lt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss4 = self.nllloss(pred1_lb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss5 = self.nllloss(pred1_rt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss6 = self.nllloss(pred1_rb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)

                coords_lt = grid.floor().float() - grid.float()
                coords_rb = grid.ceil().float() - grid.float()
                coords_lb = torch.cat((coords_rb[:,0,:,:].unsqueeze(dim=1), coords_lt[:,1,:,:].unsqueeze(dim=1)), 1) # coords_lt[..., 0] : row | coords_lt[..., 1] : col
                coords_rt = torch.cat((coords_lt[:,0,:,:].unsqueeze(dim=1), coords_rb[:,1,:,:].unsqueeze(dim=1)), 1)

                gt_offsets = torch.zeros(offsets.shape).to(offsets.device)
                gt_offsets = gt_offsets + offsets
                min_error = pred1_loss1

                error_map = torch.lt(pred1_loss3, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lt-gt_offsets)
                min_error = torch.min(pred1_loss3, min_error)

                error_map = torch.lt(pred1_loss4, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lb-gt_offsets)
                min_error = torch.min(pred1_loss4, min_error)

                error_map = torch.lt(pred1_loss5, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rt-gt_offsets)
                min_error = torch.min(pred1_loss5, min_error)

                error_map = torch.lt(pred1_loss6, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rb-gt_offsets)
                min_error = torch.min(pred1_loss6, min_error)

                error_map_loss1 = torch.gt(pred1_loss1, min_error).float()
                error_map_loss1 = error_map_loss1.mul(self.offset_weight)
                error_map_loss1.add_(1.0)

            pred1_loss1.mul_(error_map_loss1.detach())
            
            offset_loss = self.smoothl1(gt_offsets.detach(), offsets)

            loss1 = torch.mean(pred1_loss1)
            loss2 = self.crossentropy(pred2, target)
            loss4 = offset_loss

            return loss1 + self.aux_weight * loss2 + self.location_regression_weight * loss4 
        elif not self.aux:
            pred1_diffdup, se_pred, pred1_detup, grid, pred1_lt_detup, pred1_lb_detup, pred1_rt_detup, pred1_rb_detup, offsets, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
            
            with torch.no_grad():
                pred1_detup_logsoftmax  = self.logsoftmax(pred1_detup)
                pred1_lt_detup_logsoftmax = self.logsoftmax(pred1_lt_detup)
                pred1_lb_detup_logsoftmax = self.logsoftmax(pred1_lb_detup)
                pred1_rt_detup_logsoftmax = self.logsoftmax(pred1_rt_detup)
                pred1_rb_detup_logsoftmax = self.logsoftmax(pred1_rb_detup)

                pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss3 = self.nllloss(pred1_lt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss4 = self.nllloss(pred1_lb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss5 = self.nllloss(pred1_rt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss6 = self.nllloss(pred1_rb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)

                coords_lt = grid.floor().float() - grid.float()
                coords_rb = grid.ceil().float() - grid.float()
                coords_lb = torch.cat((coords_rb[:,0,:,:].unsqueeze(dim=1), coords_lt[:,1,:,:].unsqueeze(dim=1)), 1) # coords_lt[..., 0] : row | coords_lt[..., 1] : col
                coords_rt = torch.cat((coords_lt[:,0,:,:].unsqueeze(dim=1), coords_rb[:,1,:,:].unsqueeze(dim=1)), 1)

                gt_offsets = torch.zeros(offsets.shape).to(offsets.device)
                gt_offsets = gt_offsets + offsets
                min_error = pred1_loss1

                error_map = torch.lt(pred1_loss3, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lt-gt_offsets)
                min_error = torch.min(pred1_loss3, min_error)

                error_map = torch.lt(pred1_loss4, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lb-gt_offsets)
                min_error = torch.min(pred1_loss4, min_error)

                error_map = torch.lt(pred1_loss5, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rt-gt_offsets)
                min_error = torch.min(pred1_loss5, min_error)

                error_map = torch.lt(pred1_loss6, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rb-gt_offsets)
                min_error = torch.min(pred1_loss6, min_error)

                error_map_loss1 = torch.gt(pred1_loss1, min_error).float()
                error_map_loss1 = error_map_loss1.mul(self.offset_weight)
                error_map_loss1.add_(1.0)

            pred1_loss1.mul_(error_map_loss1.detach())
            
            offset_loss = self.smoothl1(gt_offsets.detach(), offsets)

            loss1 = torch.mean(pred1_loss1)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            loss4 = offset_loss

            return loss1 + self.se_weight * loss3 + self.location_regression_weight * loss4 
        else:
            pred1_diffdup, pred1_detup, grid, pred1_lt_detup, pred1_lb_detup, pred1_rt_detup, pred1_rb_detup, offsets, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
            
            with torch.no_grad():
                pred1_detup_logsoftmax  = self.logsoftmax(pred1_detup)
                pred1_lt_detup_logsoftmax = self.logsoftmax(pred1_lt_detup)
                pred1_lb_detup_logsoftmax = self.logsoftmax(pred1_lb_detup)
                pred1_rt_detup_logsoftmax = self.logsoftmax(pred1_rt_detup)
                pred1_rb_detup_logsoftmax = self.logsoftmax(pred1_rb_detup)

                pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss3 = self.nllloss(pred1_lt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss4 = self.nllloss(pred1_lb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss5 = self.nllloss(pred1_rt_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)
                pred1_loss6 = self.nllloss(pred1_rb_detup_logsoftmax, target_1.squeeze().long()).unsqueeze(dim=1)

                coords_lt = grid.floor().float() - grid.float()
                coords_rb = grid.ceil().float() - grid.float()
                coords_lb = torch.cat((coords_rb[:,0,:,:].unsqueeze(dim=1), coords_lt[:,1,:,:].unsqueeze(dim=1)), 1) # coords_lt[..., 0] : row | coords_lt[..., 1] : col
                coords_rt = torch.cat((coords_lt[:,0,:,:].unsqueeze(dim=1), coords_rb[:,1,:,:].unsqueeze(dim=1)), 1)

                gt_offsets = torch.zeros(offsets.shape).to(offsets.device)
                gt_offsets = gt_offsets + offsets
                min_error = pred1_loss1

                error_map = torch.lt(pred1_loss3, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lt-gt_offsets)
                min_error = torch.min(pred1_loss3, min_error)

                error_map = torch.lt(pred1_loss4, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_lb-gt_offsets)
                min_error = torch.min(pred1_loss4, min_error)

                error_map = torch.lt(pred1_loss5, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rt-gt_offsets)
                min_error = torch.min(pred1_loss5, min_error)

                error_map = torch.lt(pred1_loss6, min_error).float()
                gt_offsets = gt_offsets + error_map*(coords_rb-gt_offsets)
                min_error = torch.min(pred1_loss6, min_error)

                error_map_loss1 = torch.gt(pred1_loss1, min_error).float()
                error_map_loss1 = error_map_loss1.mul(self.offset_weight)
                error_map_loss1.add_(1.0)

            pred1_loss1.mul_(error_map_loss1.detach())
            
            offset_loss = self.smoothl1(gt_offsets.detach(), offsets)

            loss1 = torch.mean(pred1_loss1)
            loss4 = offset_loss

            return loss1 + self.location_regression_weight * loss4 
            
    @staticmethod
    def to_one_hot(labels, C=2):
        one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3)).cuda().to(labels.device)
        target = one_hot.scatter_(1, labels.long(), 1.0)
        return target

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
'''        
class OffsetLosses(Module):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, offset=True, offset_weight=0.5, weight=None,
                 size_average=True, ignore_index=-1):
        super(OffsetLosses, self).__init__()
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.offset = offset
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.offset_weight = offset_weight
        self.bceloss = BCELoss(weight, size_average)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss(reduction='none', ignore_index=ignore_index)
        # self.crossentropy = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)

    def forward(self, *inputs):
        if self.se_loss and self.aux:
            pred1_diffdup, se_pred, pred1_detup, pred2_detup, pred2_diffup, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            pred1_detup_logsoftmax = self.logsoftmax(pred1_detup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            # pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target)
            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long())
            pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long())

            error_map1 = torch.gt(pred1_loss1, pred1_loss2).float()
            error_map1.mul_(self.offset_weight)
            error_map1.add_(1.0)
            pred1_loss1.mul_(error_map1.detach())

            loss1 = torch.mean(pred1_loss1)

            pred2_diffup_logsoftmax = self.logsoftmax(pred2_diffup)
            pred2_detup_logsoftmax = self.logsoftmax(pred2_detup)
            
            target_2 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred2_diffup.size(2),pred2_diffup.size(3)), mode='nearest')

            # pred2_loss1 = self.nllloss(pred2_diffup_logsoftmax, target)
            pred2_loss1 = self.nllloss(pred2_diffup_logsoftmax, target_2.squeeze().long())
            pred2_loss2 = self.nllloss(pred2_detup_logsoftmax, target_2.squeeze().long())

            error_map2 = torch.gt(pred2_loss1, pred2_loss2).float()
            error_map2.mul_(self.offset_weight)
            error_map2.add_(1.0)
            pred2_loss1.mul_(error_map2.detach())

            loss2 = torch.mean(pred2_loss1)

            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3
        elif not self.se_loss:
            pred1_diffdup, pred1_detup, pred2_detup, pred2_diffup, target = tuple(inputs)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            pred1_detup_logsoftmax = self.logsoftmax(pred1_detup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long())
            pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long())

            error_map1 = torch.gt(pred1_loss1, pred1_loss2).float()
            error_map1.mul_(self.offset_weight)
            error_map1.add_(1.0)
            pred1_loss1.mul_(error_map1.detach())

            loss1 = torch.mean(pred1_loss1)

            pred2_diffup_logsoftmax = self.logsoftmax(pred2_diffup)
            pred2_detup_logsoftmax = self.logsoftmax(pred2_detup)
            
            target_2 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred2_diffup.size(2),pred2_diffup.size(3)), mode='nearest')

            pred2_loss1 = self.nllloss(pred2_diffup_logsoftmax, target_2.squeeze().long())
            pred2_loss2 = self.nllloss(pred2_detup_logsoftmax, target_2.squeeze().long())

            error_map2 = torch.gt(pred2_loss1, pred2_loss2).float()
            error_map2.mul_(self.offset_weight)
            error_map2.add_(1.0)
            pred2_loss1.mul_(error_map2.detach())

            loss2 = torch.mean(pred2_loss1)

            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred1_diffdup, se_pred, pred1_detup, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1_diffdup)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            pred1_detup_logsoftmax = self.logsoftmax(pred1_detup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long())
            pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long())

            error_map1 = torch.gt(pred1_loss1, pred1_loss2).float()
            error_map1.mul_(self.offset_weight)
            error_map1.add_(1.0)
            pred1_loss1.mul_(error_map1.detach())

            loss1 = torch.mean(pred1_loss1)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss3
        else:
            pred1_diffdup, pred1_detup, target = tuple(inputs)

            pred1_diffup_logsoftmax = self.logsoftmax(pred1_diffdup)
            pred1_detup_logsoftmax = self.logsoftmax(pred1_detup)
            
            target_1 = F.interpolate(target.unsqueeze(dim=1).float(), size=(pred1_diffdup.size(2),pred1_diffdup.size(3)), mode='nearest')

            pred1_loss1 = self.nllloss(pred1_diffup_logsoftmax, target_1.squeeze().long())
            pred1_loss2 = self.nllloss(pred1_detup_logsoftmax, target_1.squeeze().long())

            error_map1 = torch.gt(pred1_loss1, pred1_loss2).float()
            error_map1.mul_(self.offset_weight)
            error_map1.add_(1.0)
            pred1_loss1.mul_(error_map1.detach())

            loss1 = torch.mean(pred1_loss1)
            return loss1
            
    @staticmethod
    def to_one_hot(labels, C=2):
        one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3)).cuda().to(labels.device)
        target = one_hot.scatter_(1, labels.long(), 1.0)
        return target

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
'''
class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat

class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)
