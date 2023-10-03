# Copyright (c) Phigent Robotics. All rights reserved.

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from functools import partial
import numpy as np
from copy import deepcopy
from functools import partial
import cv2
from PIL import Image

import mmcv
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint, load_state_dict, force_fp32
from mmdet.models import DETECTORS, build_detector
from mmdet.utils import get_root_logger
from .centerpoint import CenterPoint
from .bevdet import BEVDet
from ..bricks import NonLocalBlockND
from mmdet3d.core import centerpoint_radius_func1, centerpoint_radius_func2, centerpoint_radius_func3, maxwh_radius_func
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import multi_apply
from mmdet3d.models.builder import build_loss
from mmdet3d.core.points import LiDARPoints
from mmdet3d.core.bbox import LiDARInstance3DBoxes, box_np_ops
from .bevdet import BEVDepth4D, BEVDetSequentialES
from .centerpoint import CenterPoint
import time


def dist2(tensor_a, tensor_b, mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    # diff = diff * attention_mask
    # diff = diff * channel_attention_mask
    if mask is not None:
        diff = diff * mask
    # diff = torch.sum(diff) ** 0.5
    diff = torch.sum(diff, dim=tuple(range(1, diff.dim()))) ** 0.5
    diff = diff.mean()
    return diff


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=partial(nn.ReLU, inplace=True), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.stride = _pair(1)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # FIXME to accommodate NCHW tensor
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TwoLayer(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=partial(nn.ReLU, inplace=True),
                 norm_layer=nn.BatchNorm2d, kernel_size=4, stride=4, padding=0):
        super().__init__()
        assert isinstance(norm_layer, dict) or isinstance(norm_layer(1), nn.BatchNorm2d )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.stride = _pair(stride)
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_layer(hidden_features) if not isinstance(norm_layer, dict) else \
            build_norm_layer(norm_layer, hidden_features)[1]
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(out_features) if not isinstance(norm_layer, dict) else \
            build_norm_layer(norm_layer, out_features)[1]
        self.act2 = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class ThreeLayer(nn.Module):
    """ add one more conv bn relu based on Two Layer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=partial(nn.ReLU, inplace=True),
                 norm_layer=nn.BatchNorm2d, kernel_size=4, stride=4, padding=0):
        super().__init__()
        assert isinstance(norm_layer, dict) or isinstance(norm_layer(1), nn.BatchNorm2d )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.stride = _pair(stride)
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_layer(hidden_features) if not isinstance(norm_layer, dict) else \
            build_norm_layer(norm_layer, hidden_features)[1]
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(hidden_features) if not isinstance(norm_layer, dict) else \
            build_norm_layer(norm_layer, hidden_features)[1]
        self.act2 = act_layer()
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.norm3 = norm_layer(out_features) if not isinstance(norm_layer, dict) else \
            build_norm_layer(norm_layer, out_features)[1]
        self.act3 = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)
        return x


def draw_scale(scale_mask, center, shape, radius):
    x, y = int(center[0]), int(center[1])
    length, width = shape
    area = torch.ceil(length * width)
    value = 1.0 / area

    # below is modified from draw_heatmap_gaussian
    height, width = scale_mask.shape[0:2]

    # these lrtb are the distance of center point to four edges
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_scale_mask = scale_mask[y - top:y + bottom, x - left:x + right]
    # maximum
    masked_scale_mask[masked_scale_mask < value] = value
    return masked_scale_mask


@DETECTORS.register_module()
class BEVDetDistill(BEVDet):
    def __init__(self, teacher_config, teacher_ckpt, distill_type, distill_params, eval_teacher=True, self_ckpt=None,
                 inherit_head=False, **kwargs):
        super(BEVDetDistill, self).__init__(**kwargs)
        self.eval_teacher = eval_teacher
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if isinstance(teacher_ckpt, str) and teacher_ckpt.lower() != 'none':
            print(f'loading teacher checkpoint from {teacher_ckpt}')
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        # # FIXME for now, only allow centerpoint as teacher
        # assert isinstance(self.teacher_model, CenterPoint) and not isinstance(self.teacher_model, BEVDet)

        # TODO: inherit
        if isinstance(self_ckpt, str) and self_ckpt.lower() != 'none':
            print(f'loading pretrained bevdet distill checkpoint')
            load_checkpoint(self, self_ckpt, map_location='cpu')

        self.inherit_head = inherit_head
        if self.inherit_head:
            assert isinstance(teacher_ckpt, str) and teacher_ckpt.lower() != 'none'

        self.distill_type = distill_type
        self.distill_params = distill_params
        assert distill_type in ['all', 'foreground_background', 'linfengzhang', 's2m2_ssd_heatmap', 's2m2_ssd_feature',
                                'gauss_focal_heatmap', 'non_local', 'affinity', 'fgd']
        assert 'heatmap' not in distill_type, 'to add heatmap distill support in the future!'

        if distill_type not in ['s2m2_ssd_heatmap', 'affinity', 'gauss_focal_heatmap', 'fgd']:
            self.adaptation_type = distill_params['adaptation_type']
            student_channels, teacher_channels = distill_params['student_channels'], distill_params['teacher_channels']
            # according to the iclr2021 paper's code, 1x1conv is the default choice. But still need to ablate on 3d detection
            if self.adaptation_type == '3x3conv':
                self.adaptation_layers = nn.ModuleList([
                    nn.Conv2d(student_channel, teacher_channel,
                              kernel_size=3, stride=1, padding=1)
                    for student_channel, teacher_channel in zip(student_channels, teacher_channels)])
            elif self.adaptation_type == '1x1conv':
                #   1x1 conv
                self.adaptation_layers = nn.ModuleList([
                    nn.Conv2d(student_channel, teacher_channel,
                              kernel_size=1, stride=1, padding=0)
                    for student_channel, teacher_channel in zip(student_channels, teacher_channels)])
            elif self.adaptation_type == 'mlp':
                self.adaptation_layers = nn.ModuleList([
                    Mlp(in_features=student_channel, out_features=teacher_channel)
                    for student_channel, teacher_channel in zip(student_channels, teacher_channels)])
            else:
                raise NotImplementedError

        if distill_type == 'linfengzhang':
            student_channels, teacher_channels = distill_params['student_channels'], distill_params['teacher_channels']
            self.channel_wise_adaptations = nn.ModuleList([
                nn.Linear(student_channel, teacher_channel)
                for student_channel, teacher_channel in zip(student_channels, teacher_channels)])
            self.spatial_wise_adaptations = nn.ModuleList([
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
                for student_channel, teacher_channel in zip(student_channels, teacher_channels)])

        if distill_type == 'fgd':
            student_channels, teacher_channels = distill_params['student_channels'], distill_params['teacher_channels']
            if isinstance(self.distill_params['affinity_mode'], str):
                self.distill_params['affinity_mode'] = [self.distill_params['affinity_mode'] for _ in student_channels]
            if isinstance(self.distill_params['fp_as_foreground'], str):
                self.distill_params['fp_as_foreground'] = [self.distill_params['fp_as_foreground'] for _ in student_channels]
            assert len(student_channels) == len(teacher_channels)
            if isinstance(distill_params['adaptation_type'], str):
                distill_params['adaptation_type'] = [distill_params['adaptation_type'] for _ in student_channels]
            if isinstance(distill_params['teacher_adaptation_type'], str):
                distill_params['teacher_adaptation_type'] = [distill_params['teacher_adaptation_type'] for _ in student_channels]
            self.channel_wise_adaptations = []
            self.teacher_adaptations = []
            for index, (adaptation_type, teacher_adaptation_type, student_channel, teacher_channel) in \
                    enumerate(zip(distill_params['adaptation_type'], distill_params['teacher_adaptation_type'],
                                  student_channels, teacher_channels)):
                if adaptation_type == '1x1conv':
                    self.channel_wise_adaptations.append(nn.Conv2d(student_channel, teacher_channel,
                                  kernel_size=1, stride=1, padding=0))
                elif adaptation_type == '3x3conv':
                    self.channel_wise_adaptations.append(nn.Conv2d(student_channel, teacher_channel,
                                  kernel_size=3, stride=1, padding=1))
                elif adaptation_type == 'mlp':
                    self.channel_wise_adaptations.append(Mlp(in_features=student_channel, out_features=teacher_channel))
                elif adaptation_type == '2layer':
                    self.channel_wise_adaptations.append(TwoLayer(in_features=student_channel, out_features=teacher_channel,
                                 kernel_size=self.distill_params['student_adaptation_params']['kernel_size'],
                                 stride=self.distill_params['student_adaptation_params']['stride']))
                    assert self.distill_params['student_adaptation_params']['kernel_size'] == 1 # use downsample_2layer for >1 stride
                    assert self.distill_params['student_adaptation_params']['stride'] == 1
                elif adaptation_type == '3layer':
                    self.channel_wise_adaptations.append(ThreeLayer(in_features=student_channel, out_features=teacher_channel,
                                 kernel_size=self.distill_params['student_adaptation_params']['kernel_size'],
                                 stride=self.distill_params['student_adaptation_params']['stride']))
                    assert self.distill_params['student_adaptation_params']['kernel_size'] == 1 # use downsample_2layer for >1 stride
                    assert self.distill_params['student_adaptation_params']['stride'] == 1
                elif adaptation_type == 'downsample_2layer':
                    self.channel_wise_adaptations.append(
                        TwoLayer(in_features=student_channel, out_features=teacher_channel,
                                 kernel_size=self.distill_params['student_adaptation_params']['downsample_kernel_size'],
                                 stride=self.distill_params['student_adaptation_params']['downsample_stride'])
                    )
                elif adaptation_type == 'identity':
                    self.channel_wise_adaptations.append(nn.Identity())
                    self.channel_wise_adaptations[index].stride = _pair(1)
                elif adaptation_type == 'upsample_2layer':
                    self.channel_wise_adaptations.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=self.distill_params['student_adaptation_params']['upsample_factor'],
                                        mode='bilinear', align_corners=True),
                            TwoLayer(in_features=student_channel, out_features=teacher_channel,
                                     kernel_size=self.distill_params['student_adaptation_params']['kernel_size'],
                                     stride=self.distill_params['student_adaptation_params']['stride'])
                        )
                    )
                    assert self.distill_params['student_adaptation_params']['upsample_factor'] % self.distill_params['student_adaptation_params']['stride']  == 0
                    assert self.distill_params['student_adaptation_params']['stride'] == 1
                    self.channel_wise_adaptations[index].stride = _pair(self.distill_params['student_adaptation_params']['stride']
                                                 / self.distill_params['student_adaptation_params']['upsample_factor'])
                elif adaptation_type == 'upsample_3layer':
                    self.channel_wise_adaptations.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=self.distill_params['student_adaptation_params']['upsample_factor'],
                                        mode='bilinear', align_corners=True),
                            ThreeLayer(in_features=student_channel, out_features=teacher_channel,
                                     kernel_size=self.distill_params['student_adaptation_params']['kernel_size'],
                                     stride=self.distill_params['student_adaptation_params']['stride'])
                        )
                    )
                    assert self.distill_params['student_adaptation_params']['upsample_factor'] % self.distill_params['student_adaptation_params']['stride']  == 0
                    assert self.distill_params['student_adaptation_params']['stride'] == 1
                    self.channel_wise_adaptations[index].stride = _pair(self.distill_params['student_adaptation_params']['stride']
                                                 / self.distill_params['student_adaptation_params']['upsample_factor'])
                elif adaptation_type == 'upsample_1x1conv':
                    self.channel_wise_adaptations.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=self.distill_params['student_adaptation_params']['upsample_factor'],
                                        mode='bilinear', align_corners=True),
                            nn.Conv2d(student_channel, teacher_channel,
                                      kernel_size=1, stride=1, padding=0)
                        )
                    )
                    assert self.distill_params['student_adaptation_params']['upsample_factor'] % self.distill_params['student_adaptation_params']['stride']  == 0
                    assert self.distill_params['student_adaptation_params']['stride'] == 1
                    self.channel_wise_adaptations[index].stride = _pair(self.distill_params['student_adaptation_params']['stride']
                                                 / self.distill_params['student_adaptation_params']['upsample_factor'])
                elif adaptation_type == 'avgpool_1x1conv':
                    self.channel_wise_adaptations.append(
                        nn.Sequential(
                            nn.AvgPool2d(kernel_size=self.distill_params['student_adaptation_params']['downsample_kernel_size']),
                            nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0)
                        )
                    )
                    self.channel_wise_adaptations[index].stride = _pair(self.distill_params['student_adaptation_params']['downsample_kernel_size'])
                else:
                    print(f'distill_params[adaptation_type]')
                    print(adaptation_type)
                    raise NotImplementedError

                # assert distill_params['teacher_adaptation_type'] == 'identity'
                # if distill_params['teacher_adaptation_type'] == '2layer':
                #     self.teacher_adaptations.append(
                #         TwoLayer(in_features=teacher_channel, out_features=teacher_channel,
                #                  kernel_size=self.distill_params['teacher_adaptation_params']['kernel_size'],
                #                  stride=self.distill_params['teacher_adaptation_params']['stride'])
                #     )
                # if distill_params['teacher_adaptation_type'] == '2layer_s_channel':
                #     self.teacher_adaptations = nn.ModuleList([
                #         TwoLayer(in_features=teacher_channel, out_features=student_channel,
                #                  kernel_size=self.distill_params['teacher_adaptation_params']['kernel_size'],
                #                  stride=self.distill_params['teacher_adaptation_params']['stride'])
                #         for student_channel, teacher_channel in zip(student_channels, teacher_channels)])
                if teacher_adaptation_type == 'avgpool':
                    self.teacher_adaptations.append(nn.AvgPool2d(**self.distill_params['teacher_adaptation_params']))
                    self.teacher_adaptations[index].stride = _pair(self.teacher_adaptations[index].stride)
                elif teacher_adaptation_type == 'maxpool':
                    self.teacher_adaptations.append(nn.MaxPool2d(**self.distill_params['teacher_adaptation_params']))
                    self.teacher_adaptations[index].stride = _pair(self.teacher_adaptations[index].stride)
                elif teacher_adaptation_type == 'identity':
                    self.teacher_adaptations.append(nn.Identity())
                    self.teacher_adaptations[index].stride = _pair(1)
                elif teacher_adaptation_type == 'downsample_3layer':
                    self.teacher_adaptations.append(ThreeLayer(in_features=teacher_channel, out_features=student_channel, kernel_size=self.distill_params['teacher_adaptation_params']['kernel_size'], stride=self.distill_params['teacher_adaptation_params']['stride']))
                elif teacher_adaptation_type == 'avgpool_3layer':
                    self.teacher_adaptations.append(nn.AvgPool2d(**self.distill_params['teacher_adaptation_params']))
                    self.teacher_adaptations[index].stride = _pair(self.teacher_adaptations[index].stride)
                    self.teacher_adaptations.append(ThreeLayer(in_features=teacher_channel, out_features=student_channel, kernel_size=1, stride=1))
                else:
                    raise NotImplementedError

            self.channel_wise_adaptations = nn.ModuleList(self.channel_wise_adaptations)
            self.teacher_adaptations = nn.ModuleList(self.teacher_adaptations)
            if distill_params['spatial_mask']:
                self.spatial_wise_adaptations = nn.ModuleList([
                    nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
                    for student_channel, teacher_channel in zip(student_channels, teacher_channels)])

        if distill_type == 'non_local':
            student_channels, teacher_channels = distill_params['student_channel'], distill_params['teacher_channel']
            self.student_non_locals = nn.ModuleList([NonLocalBlockND(in_channels=student_channel)
                                                    for student_channel in student_channels])
            self.teacher_non_locals = nn.ModuleList([NonLocalBlockND(in_channels=teacher_channel)
                                                    for teacher_channel in teacher_channels])

        # if self.distill_type == 's2m2_ssd_feature':
        self.count = 0
        self.count_thres = 25

        # self._epoch = -1
        self._epoch = 1

    def init_weights(self):
        super(BEVDetDistill, self).init_weights()
        if self.inherit_head:
            logger = get_root_logger()
            logger.info(f'inherit teacher head')
            load_state_dict(self.pts_bbox_head.task_heads, self.teacher_model.pts_bbox_head.task_heads.state_dict(),
                            strict=False, logger=logger)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def linfengzhang_distill_loss(self, teacher_feat, student_feat, index):
        '''
        # ICLR2021 paper IMPROVE OBJECT DETECTION WITH FEATURE-BASED KNOWLEDGE DISTILLATION: TOWARDS ACCURATE AND EFFICIENT DETECTORS
        Args:
            teacher_feat: BCHW tensor
            student_feat: BCHW tensor

        Returns: a list of knowledge distillation losses

        '''
        # for spatial attention
        S_T = self.distill_params['spatial_t'] # 0.1
        s_ratio = self.distill_params['spatial_student_ratio'] # 1.0
        #   for channel attention
        C_T = self.distill_params['channel_t']  # 0.1
        c_s_ratio = self.distill_params['channel_student_ratio']  # 1.0
        # loss weight
        kd_feat_loss_weight = self.distill_params['feat_loss_weights'][index]
        kd_channel_loss_weight = self.distill_params['channel_loss_weights'][index]
        kd_spatial_loss_weight = self.distill_params['spatial_loss_weights'][index]
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
        loss_dict = dict()

        student_B, student_C, student_H, student_W = student_feat.size()
        teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
        assert student_B == teacher_B
        B = student_B


        t_attention_mask = torch.mean(torch.abs(teacher_feat), [1], keepdim=True)
        t_attention_mask = t_attention_mask.view(B, -1)
        t_attention_mask = torch.softmax(t_attention_mask / S_T, dim=1) * teacher_H * teacher_W
        t_attention_mask = t_attention_mask.view(teacher_B, 1, teacher_H, teacher_W)
        # t_attention_mask = torch.mean(torch.abs(teacher_feat), [1], keepdim=True)
        # t_attention_mask = t_attention_mask.view(B, -1)
        # t_attention_mask_softmax = torch.softmax(t_attention_mask / S_T, dim=1) * teacher_H * teacher_W
        # t_attention_mask_softmax = t_attention_mask_softmax.view(teacher_B, 1, teacher_H, teacher_W)

        s_attention_mask = torch.mean(torch.abs(student_feat), [1], keepdim=True)
        s_attention_mask = s_attention_mask.view(B, -1)
        s_attention_mask = torch.softmax(s_attention_mask / S_T, dim=1) * student_H * student_W
        s_attention_mask = s_attention_mask.view(student_B, 1, student_H, student_W)
        # s_attention_mask = torch.mean(torch.abs(self.spatial_wise_adaptation(student_feat)), [1], keepdim=True)
        # s_attention_mask = s_attention_mask.view(B, -1)
        # s_attention_mask_softmax = torch.softmax(s_attention_mask / S_T, dim=1) * student_H * student_W
        # s_attention_mask_softmax = s_attention_mask_softmax.view(student_B, 1, student_H, student_W)

        c_t_attention_mask = torch.mean(torch.abs(teacher_feat), [2, 3], keepdim=True)  # B x C x 1 x1
        c_t_attention_mask = c_t_attention_mask.view(B, -1)  # B x C
        c_t_attention_mask = torch.softmax(c_t_attention_mask / C_T, dim=1) * teacher_C
        c_t_attention_mask = c_t_attention_mask.view(teacher_B, teacher_C, 1, 1)  # B x C -> B x C x 1 x1
        # c_t_attention_mask = torch.mean(torch.abs(teacher_feat), [2, 3], keepdim=True)  # B x C x 1 x1
        # c_t_attention_mask = c_t_attention_mask.view(B, -1)  # B x C
        # c_t_attention_mask_softmax = torch.softmax(c_t_attention_mask / C_T, dim=1) * teacher_C
        # c_t_attention_mask_softmax = c_t_attention_mask_softmax.view(teacher_B, teacher_C, 1, 1)  # B x C -> B x C x 1 x1

        c_s_attention_mask = torch.mean(torch.abs(student_feat), [2, 3], keepdim=True)
        c_s_attention_mask = c_s_attention_mask.view(B, -1)
        c_s_attention_mask = torch.softmax(c_s_attention_mask / C_T, dim=1) * student_C
        c_s_attention_mask = c_s_attention_mask.view(student_B, student_C, 1, 1)
        # c_s_attention_mask = torch.mean(torch.abs(self.channel_wise_adaptation(student_feat)), [2, 3], keepdim=True)
        # c_s_attention_mask = c_s_attention_mask.view(B, -1)
        # c_s_attention_mask_softmax = torch.softmax(c_s_attention_mask / C_T, dim=1) * teacher_C
        # c_s_attention_mask_softmax = c_s_attention_mask_softmax.view(student_B, teacher_C, 1, 1)

        sum_attention_mask = t_attention_mask
        sum_attention_mask = sum_attention_mask.detach()
        c_sum_attention_mask = c_t_attention_mask
        c_sum_attention_mask = c_sum_attention_mask.detach()
        # sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
        # sum_attention_mask = sum_attention_mask.detach()
        # c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
        # c_sum_attention_mask = c_sum_attention_mask.detach()
        # sum_attention_mask = (t_attention_mask_softmax + s_attention_mask_softmax * s_ratio) / (1 + s_ratio)
        # sum_attention_mask = sum_attention_mask.detach()
        # c_sum_attention_mask = (c_t_attention_mask_softmax + c_s_attention_mask_softmax * c_s_ratio) / (1 + c_s_ratio)
        # c_sum_attention_mask = c_sum_attention_mask.detach()

        # FIXME it seems that the batch dim is not considered?
        kd_feat_loss = dist2(teacher_feat, self.adaptation_layers[index](student_feat),
                             mask=sum_attention_mask*c_sum_attention_mask) * kd_feat_loss_weight  # * 7e-5 * 6
        # FIXME how to project student channel to teacher channel in a natural manner?
        kd_channel_loss += torch.dist(torch.mean(teacher_feat, [2, 3]),
                                      self.channel_wise_adaptations[index](torch.mean(student_feat, [2, 3]))) \
                           * kd_spatial_loss_weight # 4e-3 * 6
        t_spatial_pool = torch.mean(teacher_feat, [1], keepdim=True).view(teacher_B, 1, teacher_H, teacher_W)
        s_spatial_pool = torch.mean(student_feat, [1], keepdim=True).view(student_B, 1, student_H, student_W)
        kd_spatial_loss += torch.dist(t_spatial_pool,
                                      self.spatial_wise_adaptations[index](s_spatial_pool)) * kd_spatial_loss_weight # 4e-3 * 6
        # kd_channel_loss = torch.dist(c_t_attention_mask, c_s_attention_mask) * kd_channel_loss_weight #* 4e-3 * 6
        # kd_spatial_loss = torch.dist(t_attention_mask, s_attention_mask) * kd_spatial_loss_weight

        loss_dict.update({'kd_feat_loss': kd_feat_loss})
        loss_dict.update({'kd_channel_loss': kd_channel_loss})
        loss_dict.update({'kd_spatial_loss': kd_spatial_loss})

        # kd_nonlocal_loss = 0
        # non_local_loss_weight = self.distill_params['non_local_loss_weight']
        # if t_info is not None:
        #     t_feats = t_info['feat']
        #     for _i in range(len(t_feats)):
        #         s_relation = self.student_non_local[_i](x[_i])
        #         t_relation = self.teacher_non_local[_i](t_feats[_i])
        #         #   print(s_relation.size())
        #         kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2) * non_local_loss_weight

        # loss_dict.update(kd_nonlocal_loss=kd_nonlocal_loss)

        return loss_dict


    def all_distill_loss(self, teacher_feat, student_feat, index):
        p = self.distill_params['p']  # 2
        kd_feat_loss_weight = self.distill_params['feat_loss_weights'][index]
        loss_dict = dict()

        if p == 1:
            loss = F.l1_loss(self.adaptation_layers[index](student_feat), teacher_feat) * kd_feat_loss_weight
        elif p == 2:
            loss = F.mse_loss(self.adaptation_layers[index](student_feat), teacher_feat) * kd_feat_loss_weight
        else:
            raise NotImplementedError
        loss_dict['kd_feat_loss'] = loss
        return loss_dict


    def s2m2_ssd_mask(self, heatmaps, student_preds):
        thres = self.distill_params['output_threshold']

        # TODO: check shape consistency
        # no need to sigmoid here
        gt_batch_hm = [heatmap for heatmap in heatmaps]
        gt_batch_hm = torch.cat(gt_batch_hm, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]

        student_batch_hm = [clip_sigmoid(student_pred_dict[0]['heatmap']) for student_pred_dict in student_preds]
        student_batch_hm = torch.cat(student_batch_hm, dim=1)
        student_batch_hm_max = torch.max(student_batch_hm, dim=1, keepdim=True)[0]
        student_batch_hm_max = student_batch_hm_max.detach()

        tp_mask = torch.logical_and(gt_batch_hm_max > thres, student_batch_hm_max > thres)
        fp_mask = torch.logical_and(gt_batch_hm_max < thres, student_batch_hm_max > thres)
        fn_mask = torch.logical_and(gt_batch_hm_max>thres, student_batch_hm_max<thres)

        return tp_mask, fp_mask, fn_mask


    def s2m2_ssd_heatmap_distill_loss(self, teacher_preds, student_preds, tp_mask, fp_mask, fn_mask):
        tp_weight = self.distill_params['tp_weight']
        fpfn_weight = self.distill_params['fpfn_weight']
        criterion = self.distill_params['criterion']
        loss_dict = dict()

        # TODO: check if need to combine all tasks heatmap here or not
        teacher_batch_hm = [clip_sigmoid(teacher_pred_dict[0]['heatmap']) for teacher_pred_dict in teacher_preds]
        teacher_batch_hm = torch.cat(teacher_batch_hm, dim=1)
        ####################
        teacher_batch_hm = torch.max(teacher_batch_hm, dim=1, keepdim=True)[0]
        ####################
        student_batch_hm = [clip_sigmoid(student_pred_dict[0]['heatmap']) for student_pred_dict in student_preds]
        student_batch_hm = torch.cat(student_batch_hm, dim=1)
        ###################
        student_batch_hm = torch.max(student_batch_hm, dim=1, keepdim=True)[0]
        ###################

        if criterion.lower() == 'l1':
            # criterion = torch.nn.L1Loss(reduction='none')
            criterion = partial(F.l1_loss, reduction='none')
        elif criterion.lower() == 'smoothl1':
            # criterion = torch.nn.SmoothL1Loss(reduction='none')
            criterion = partial(F.smooth_l1_loss, reduction='none')
        elif criterion.lower() == 'mse':
            # criterion = torch.nn.L2Loss(reduction='none')
            criterion = partial(F.mse_loss, reduction='none')
        else:
            raise NotImplementedError


        fpfn_mask = torch.logical_or(fp_mask, fn_mask)
        tp_mask = tp_mask.expand_as(student_batch_hm)
        fpfn_mask = fpfn_mask.expand_as(student_batch_hm)

        tp_loss = criterion(student_batch_hm * tp_mask, teacher_batch_hm * tp_mask).sum(
            dim=(1,2,3)) * tp_weight / torch.sum(tp_mask, dim=(1,2,3))
        tp_loss = tp_loss.mean()
        fpfn_loss = criterion(student_batch_hm * fpfn_mask, teacher_batch_hm).sum(
            dim=(1,2,3))* fpfn_weight / torch.sum(fpfn_mask, dim=(1,2,3))
        fpfn_loss = fpfn_loss.mean()
        loss_dict['s2m2_ssd_heatmap_kd_loss'] = tp_loss + fpfn_loss

        return loss_dict


    def gauss_focal_heatmap_distill_loss(self, teacher_preds, student_preds, heatmaps):
        weight = self.distill_params['heatmap_weight']
        criterion = self.distill_params['criterion']
        loss_dict = dict()
        if not hasattr(self, 'heatmap_distill_criterion'):
            self.heatmap_distill_criterion = build_loss(criterion)

        # TODO: check if need to combine all tasks heatmap here or not
        # teacher_batch_hm = [clip_sigmoid(teacher_pred_dict[0]['heatmap']) for teacher_pred_dict in teacher_preds]
        # teacher_batch_hm = torch.cat(teacher_batch_hm, dim=1)
        # ####################
        # teacher_batch_hm = torch.max(teacher_batch_hm, dim=1, keepdim=True)[0]
        # ####################
        # student_batch_hm = [clip_sigmoid(student_pred_dict[0]['heatmap']) for student_pred_dict in student_preds]
        # student_batch_hm = torch.cat(student_batch_hm, dim=1)
        # ###################
        # student_batch_hm = torch.max(student_batch_hm, dim=1, keepdim=True)[0]
        # ###################
        # num_pos = sum([heatmaps[task_id].eq(1).float().sum().item() for task_id in range(len(heatmaps))])
        # loss_heatmap = self.heatmap_distill_criterion(student_batch_hm, teacher_batch_hm, avg_factor=max(num_pos, 1))
        assert len(teacher_preds) == len(student_preds) and len(student_preds) == len(heatmaps)
        for task_id, (teacher_pred_dict, student_pred_dict) in enumerate(zip(teacher_preds, student_preds)):
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            teacher_hm = clip_sigmoid(teacher_pred_dict[0]['heatmap'])
            student_hm = clip_sigmoid(student_pred_dict[0]['heatmap'])
            loss_heatmap = self.heatmap_distill_criterion(student_hm, teacher_hm, avg_factor=max(num_pos, 1))
            loss_dict[f'task{task_id}_kd_heatmap_loss'] = loss_heatmap

        return loss_dict

    # not sure if implementing it is worth the time
    # def s2m2_ssd_reg_distill_loss(self, teacher_preds, student_preds, tp_mask, fn_mask):
    #     weight = self.distill_params['reg_loss_weight']
    #     criterion = self.distill_params['criterion']
    #     loss_dict = dict()
    #
    #     if criterion.lower() == 'l1':
    #         criterion = torch.nn.L1Loss(reduction='none')
    #     elif criterion.lower() == 'smoothl1':
    #         criterion = torch.nn.SmoothL1Loss(reduction='none')
    #     elif criterion.lower() == l2:
    #         criterion = torch.nn.L2Loss(reduction='none')
    #     else:
    #         raise NotImplementedError

    def s2m2_ssd_feature_distill_loss(self, teacher_feat, student_feat, tp_mask, fp_mask, fn_mask, index):
        tp_weight = self.distill_params['tp_weights'][index]
        fp_weight = self.distill_params['fp_weights'][index]
        fn_weight = self.distill_params['fn_weights'][index]
        criterion = self.distill_params['criterion']
        loss_dict = dict()

        if criterion.lower() == 'l1':
            # criterion = torch.nn.L1Loss(reduction='none')
            criterion = partial(F.l1_loss, reduction='none')
        elif criterion.lower() == 'smoothl1':
            # criterion = torch.nn.SmoothL1Loss(reduction='none')
            criterion = partial(F.smooth_l1_loss, reduction='none')
        elif criterion.lower() == 'mse':
            # criterion = torch.nn.L2Loss(reduction='none')
            criterion = partial(F.mse_loss, reduction='none')
        else:
            raise NotImplementedError

        if self.distill_params['mode'] == 'old':
            fpfn_weight = self.distill_params['fpfn_weights'][index]
            fpfn_mask = torch.logical_or(fp_mask, fn_mask)
            tp_mask = tp_mask.expand_as(teacher_feat)
            fpfn_mask = fpfn_mask.expand_as(teacher_feat)

            tp_loss = criterion(self.adaptation_layers[index](student_feat) * tp_mask, teacher_feat * tp_mask).sum(
                dim=(1,2,3)) * tp_weight / torch.sum(tp_mask, dim=(1,2,3))
            tp_loss = tp_loss.mean()
            fpfn_loss = criterion(self.adaptation_layers[index](student_feat) * fpfn_mask, teacher_feat * fpfn_mask).sum(
                dim=(1,2,3))* fpfn_weight / torch.sum(fpfn_mask, dim=(1,2,3))
            fpfn_loss = fpfn_loss.mean()
            loss_dict['s2m2_ssd_feature_kd_loss'] = tp_loss + fpfn_loss
        elif self.distill_params['mode'] == 'new':
            tp_mask = tp_mask.expand_as(teacher_feat)
            fp_mask = fp_mask.expand_as(teacher_feat)
            fn_mask = fn_mask.expand_as(teacher_feat)

            loss = criterion(self.adaptation_layers[index](student_feat), teacher_feat)
            tp_loss = (loss * tp_mask).sum(dim=(1,2,3)) * tp_weight / torch.sum(tp_mask, dim=(1,2,3))
            tp_loss = tp_loss[tp_loss.isnan().logical_not()].mean()
            if tp_loss.isnan().all():
                tp_loss = loss * 0
            fp_loss = (loss * fp_mask).sum(dim=(1, 2, 3)) * fp_weight / torch.sum(fp_mask, dim=(1, 2, 3))
            fp_loss = fp_loss[fp_loss.isnan().logical_not()].mean()
            if fp_loss.isnan().all():
                fp_loss = loss * 0
            fn_loss = (loss * fn_mask).sum(dim=(1, 2, 3)) * fn_weight / torch.sum(fn_mask, dim=(1, 2, 3))
            fn_loss = fn_loss[fn_loss.isnan().logical_not()].mean()
            if fn_loss.isnan().all():
                fn_loss = loss * 0

            loss_dict['s2m2_ssd_feature_kd_tp_loss'] = tp_loss
            loss_dict['s2m2_ssd_feature_kd_fp_loss'] = fp_loss
            loss_dict['s2m2_ssd_feature_kd_fn_loss'] = fn_loss
        else:
            raise NotImplementedError

        return loss_dict

    def non_local_distill_loss(self, teacher_feat, student_feat, index):
        weight = self.distill_params['nonlocal_weights'][index]
        criterion = self.distill_params['criterion']
        loss_dict = dict()

        if criterion.lower() == 'l1':
            # criterion = torch.nn.L1Loss(reduction='none')
            criterion = partial(F.l1_loss, reduction='none')
        elif criterion.lower() == 'smoothl1':
            # criterion = torch.nn.SmoothL1Loss(reduction='none')
            criterion = partial(F.smooth_l1_loss, reduction='none')
        elif criterion.lower() == 'mse':
            # criterion = torch.nn.L2Loss(reduction='none')
            criterion = partial(F.mse_loss, reduction='none')
        else:
            raise NotImplementedError

        s_relation = self.student_non_locals[index](student_feat)
        t_relation = self.teacher_non_locals[index](teacher_feat)
        #   print(s_relation.size())
        kd_nonlocal_loss = criterion(self.adaptation_layers[index](s_relation), t_relation) * weight
        loss_dict['kd_nonlocal_loss'] = kd_nonlocal_loss

        return loss_dict


    def affinity_distill_loss(self, teacher_feat, student_feat, index):
        weight = self.distill_params['affinity_weights'][index] \
            if len(self.distill_params['affinity_weights']) > 1 else self.distill_params['affinity_weights'][0]
        criterion = self.distill_params['affinity_criterion']
        criterion = build_loss(criterion)
        loss_dict = dict()

        split = getattr(self.distill_params, 'affinity_split', 1)
        assert isinstance(split, int)

        if isinstance(teacher_feat, torch.Tensor) and isinstance(student_feat, torch.Tensor):
            if teacher_feat.dim() == 4:
                B, H, W, teacher_C = teacher_feat.shape
                teacher_feat = teacher_feat.reshape(B, -1, teacher_C)
            elif teacher_feat.dim() != 3:
                raise NotImplementedError
            if teacher_feat.dim() == 4:
                B, H, W, student_C = student_feat.shape
                student_feat = student_feat.reshape(B, -1, student_C)
            elif teacher_feat.dim() != 3:
                raise NotImplementedError
            assert teacher_feat.shape[1] == student_feat.shape[1]
            # note that dim 1 is the channel dim
            kd_affinity_loss = 0
            rand_indices = torch.randperm(teacher_feat.shape[1])
            for i in range(split):
                indices = rand_indices[i::split]
                teacher_affinity = torch.bmm(teacher_feat[:, indices, :], teacher_feat[:, indices, :].permute(0, 2, 1))
                student_affinity = torch.bmm(student_feat[:, indices, :], student_feat[:, indices, :].permute(0, 2, 1))
                kd_affinity_loss += criterion(teacher_affinity, student_affinity) * weight
            kd_affinity_loss /= split
            loss_dict['kd_affinity_loss'] = kd_affinity_loss
        elif isinstance(teacher_feat, list) and isinstance(student_feat, list):
            kd_affinity_loss = 0
            for t_feat, s_feat in zip(teacher_feat, student_feat):
                assert t_feat.dim() == 2 and s_feat.dim() == 2
                assert t_feat.shape[0] == s_feat.shape[0]
                rand_indices = torch.randperm(t_feat.shape[0])
                loss = 0
                for i in range(split):
                    indices = rand_indices[i::split]
                    teacher_affinity = torch.mm(t_feat[indices], t_feat[indices].permute(1, 0))
                    student_affinity = torch.mm(s_feat[indices], s_feat[indices].permute(1, 0))
                    loss += criterion(teacher_affinity, student_affinity) * weight
                kd_affinity_loss += loss / split
            loss_dict['kd_affinity_loss'] = kd_affinity_loss
        else:
            raise NotImplementedError

        return loss_dict


    def foreground_scale_mask(self, student_H, student_W, gt_bboxes_3d, bg_extend_length=0, bg_extend_weight=0,):
        grid_size = torch.tensor(self.pts_bbox_head.train_cfg['grid_size'])
        pc_range = torch.tensor(self.pts_bbox_head.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.pts_bbox_head.train_cfg['voxel_size'])
        # FIXME adpvatie out_size_factor. For now only support H=W
        # FIXME for now, only support nuscenes: LidarPoints/bboxes. Add support for other datasets in the future
        # out_size_factor = torch.tensor(self.pts_bbox_head.train_cfg['out_size_factor'])
        assert grid_size[0] == grid_size[1] and student_W == student_H
        assert grid_size[0] % student_W == 0
        out_size_factor = grid_size[0] // student_W

        coord_xs = [i * voxel_size[0] * out_size_factor + pc_range[0] for i in range(student_W)]
        coord_ys = [i * voxel_size[1] * out_size_factor + pc_range[1] for i in range(student_H)]
        coord_xs, coord_ys = np.meshgrid(coord_xs, coord_ys, indexing='ij')
        coord_xs = coord_xs.reshape(-1, 1)
        coord_ys = coord_ys.reshape(-1, 1)
        # make z dim 0.5
        coord_zs = np.ones_like(coord_xs) * 0.5
        coords = np.hstack((coord_xs, coord_ys, coord_zs))
        assert coords.shape[0] == student_W * student_W and coords.shape[1] == 3
        feature_pixel_points = LiDARPoints(torch.tensor(coords), 3, attribute_dims=None)

        foreground_masks = []
        fg_scale_masks = []
        bg_scale_masks = []
        for boxes in gt_bboxes_3d:
            points = feature_pixel_points.coord.numpy()
            boxes = deepcopy(boxes.tensor.numpy())
            # the first three dimension marks the bottom center in LiDARInstance3DBoxes
            # unify z dim, 0 bottom center, 1 height
            boxes[:, 2] = 0
            boxes[:, 5] = 1
            mask = box_np_ops.points_in_rbbox(points, boxes) # NxM, N is the number of points (128x128), M is the number of bboxes

            foreground_mask = mask.any(axis=-1).astype(float)
            foreground_points_indices, bbox_indices = np.nonzero(mask)
            # assert np.unique(foreground_points_indices).shape == foreground_points_indices.shape
            # unique indices here is indices of foreground_points_indices
            foreground_points_indices, unique_indices = np.unique(foreground_points_indices, return_index=True)
            bbox_indices = bbox_indices[unique_indices]
            fg_scale_mask = np.zeros(student_H*student_W, dtype=float)
            if getattr(self.distill_params, 'avg_fg_scale_mask', False):
                fg_scale_mask[foreground_points_indices] = 1.0 / (min(np.sum(foreground_mask!=0), 1))
            else:
                fg_scale_mask[foreground_points_indices] = torch.sqrt((voxel_size[0] * voxel_size[1] * out_size_factor * out_size_factor) / \
                                                        (boxes[bbox_indices][:, 3] * boxes[bbox_indices][:, 4]))

            if bg_extend_length > 0 and bg_extend_weight > 0:
                enlarged_boxes = deepcopy(boxes)
                enlarged_boxes[:, 3] = enlarged_boxes[:, 3] + (voxel_size[0] * out_size_factor * bg_extend_length).item()
                enlarged_boxes[:, 4] = enlarged_boxes[:, 4] + (voxel_size[0] * out_size_factor * bg_extend_length).item()
                enlarged_mask = box_np_ops.points_in_rbbox(points, enlarged_boxes)
                enlarged_foreground_mask = enlarged_mask.any(axis=-1)
                enlarged_foreground_points_indices, enlarged_bbox_indices = np.nonzero(enlarged_mask)
                enlarged_foreground_points_indices, unique_enlarged_indices = np.unique(enlarged_foreground_points_indices, return_index=True)
                enlarged_bbox_indices = enlarged_bbox_indices[unique_enlarged_indices]

                enlarged_foreground_mask = enlarged_foreground_mask.astype(float) * bg_extend_weight
                foreground_mask = np.maximum(foreground_mask, enlarged_foreground_mask)
                fg_scale_mask[enlarged_foreground_points_indices] = (voxel_size[0] * voxel_size[1] * out_size_factor * out_size_factor) / \
                                                    (boxes[enlarged_bbox_indices][:, 3] * boxes[enlarged_bbox_indices][:, 4])

            # use separate background scale_mask
            # assert np.equal(np.nonzero(foreground_mask), np.nonzero(scale_mask)).all()
            # all_points_indices = np.arange(student_H*student_W)
            # background_points_indices = np.setdiff1d(all_points_indices, foreground_points_indices)
            # backgound_points_number = student_H * student_W - np.sum(foreground_mask)
            # scale_mask[background_points_indices] = 1.0 / backgound_points_number
            bg_scale_mask = np.zeros(student_H * student_W, dtype=float)
            bg_points_number = student_H * student_W - np.sum(foreground_mask!=0)
            bg_scale_mask[:] = 1.0 / bg_points_number

            # FIXME check if need to transpose
            if not self.distill_params['transpose_mask']:
                foreground_mask = foreground_mask.reshape(student_W, student_H).transpose().reshape(1,1,student_H,student_W)
                fg_scale_mask = fg_scale_mask.reshape(student_W, student_H).transpose().reshape(1,1,student_H,student_W)
                bg_scale_mask = bg_scale_mask.reshape(student_W, student_H).transpose().reshape(1,1,student_H,student_W)
            else:
                foreground_mask = foreground_mask.reshape(1, 1, student_H, student_W)
                fg_scale_mask = fg_scale_mask.reshape(1, 1, student_H, student_W)
                bg_scale_mask = bg_scale_mask.reshape(1, 1, student_H, student_W)
            foreground_masks.append(torch.tensor(foreground_mask))
            fg_scale_masks.append(torch.tensor(fg_scale_mask).float())
            bg_scale_masks.append(torch.tensor(bg_scale_mask).float())
        foreground_mask = torch.cat(foreground_masks, dim=0).float() # Nx1xHxW
        fg_scale_mask = torch.cat(fg_scale_masks, dim=0)
        bg_scale_mask = torch.cat(bg_scale_masks, dim=0)

        return foreground_mask, fg_scale_mask, bg_scale_mask


    def add_fp_as_fg(self, mode, fg_mask, heatmaps, teacher_preds, student_preds):
        thres = self.distill_params['output_threshold']
        gt_thres = self.distill_params['groundtruth_threshold']
        if gt_thres is None:
            gt_thres = thres

        gt_batch_hm = [heatmap for heatmap in heatmaps]
        gt_batch_hm = torch.cat(gt_batch_hm, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]

        teacher_batch_hm = [clip_sigmoid(teacher_pred_dict[0]['heatmap']) for teacher_pred_dict in teacher_preds]
        teacher_batch_hm = torch.cat(teacher_batch_hm, dim=1)
        teacher_batch_hm_max = torch.max(teacher_batch_hm, dim=1, keepdim=True)[0]
        teacher_batch_hm_max = teacher_batch_hm_max.detach()

        # FIXME in centerpoint head loss function, student heatmaps has already been clip_sigomided
        # FIXME a easy mistake to make is to do sigmoid again
        # student_batch_hm = [clip_sigmoid(student_pred_dict[0]['heatmap']) for student_pred_dict in student_preds]
        student_batch_hm = [student_pred_dict[0]['heatmap'] for student_pred_dict in student_preds]
        student_batch_hm = torch.cat(student_batch_hm, dim=1)
        student_batch_hm_max = torch.max(student_batch_hm, dim=1, keepdim=True)[0]
        student_batch_hm_max = student_batch_hm_max.detach()

        assert teacher_batch_hm_max.shape[2] == teacher_batch_hm_max.shape[3] and student_batch_hm_max.shape[2] == student_batch_hm_max.shape[3]
        if student_batch_hm_max.shape[2] > teacher_batch_hm_max.shape[2]:
            assert student_batch_hm_max.shape[2] % teacher_batch_hm_max.shape[2] == 0
            student_batch_hm_max = F.max_pool2d(student_batch_hm_max, kernel_size=student_batch_hm_max.shape[2] // teacher_batch_hm_max.shape[2],
                                   stride=student_batch_hm_max.shape[2] // teacher_batch_hm_max.shape[2])
            gt_batch_hm_max = F.max_pool2d(gt_batch_hm_max, kernel_size=gt_batch_hm_max.shape[2] // teacher_batch_hm_max.shape[2],
                                   stride=gt_batch_hm_max.shape[2] // teacher_batch_hm_max.shape[2])
        elif student_batch_hm_max.shape[2] < teacher_batch_hm_max.shape[2]:
            assert teacher_batch_hm_max.shape[2] % student_batch_hm_max.shape[2] == 0
            student_batch_hm_max = torch.repeat_interleave(student_batch_hm_max,
                                                           repeats=teacher_batch_hm_max.shape[2] // student_batch_hm_max.shape[2], dim=2)
            student_batch_hm_max = torch.repeat_interleave(student_batch_hm_max,
                                                           repeats=teacher_batch_hm_max.shape[2] // student_batch_hm_max.shape[2], dim=3)
            gt_batch_hm_max = torch.repeat_interleave(gt_batch_hm_max,
                                                           repeats=teacher_batch_hm_max.shape[2] //
                                                                   gt_batch_hm_max.shape[2], dim=2)
            gt_batch_hm_max = torch.repeat_interleave(gt_batch_hm_max,
                                                           repeats=teacher_batch_hm_max.shape[2] //
                                                                   gt_batch_hm_max.shape[2], dim=3)

        # TODO: separate gt thres and model thres
        if mode == 'teacher':
            fp_mask = torch.logical_and(gt_batch_hm_max < gt_thres, teacher_batch_hm_max > thres)
        elif mode == 'student':
            fp_mask = torch.logical_and(gt_batch_hm_max < gt_thres, student_batch_hm_max > thres)
        elif mode == 'teacher_selected_student':
            fp_mask = torch.logical_and(gt_batch_hm_max < gt_thres, student_batch_hm_max > thres)
            fp_mask = torch.logical_and(teacher_batch_hm_max < gt_thres, fp_mask)
        elif mode == 'teacher+teacher_selected_student':
            fp_mask1 = torch.logical_and(gt_batch_hm_max < gt_thres, teacher_batch_hm_max > thres)
            fp_mask2 = torch.logical_and(gt_batch_hm_max < gt_thres, student_batch_hm_max > thres)
            fp_mask2 = torch.logical_and(teacher_batch_hm_max < gt_thres, fp_mask2)
            fp_mask = torch.logical_or(fp_mask1, fp_mask2)
        else:
            raise NotImplementedError

        # don't change non-zero foreground points
        assert fp_mask.shape[2] == fp_mask.shape[3] and fg_mask.shape[2] == fg_mask.shape[3]
        if fp_mask.shape[2] > fg_mask.shape[2]:
            assert fp_mask.shape[2] % fg_mask.shape[2] == 0
            try:
                fp_mask = F.max_pool2d(fp_mask, kernel_size=fp_mask.shape[2]//fg_mask.shape[2],
                                       stride=fp_mask.shape[2]//fg_mask.shape[2])
            except:
                fp_mask = F.max_pool2d(fp_mask.float(), kernel_size=fp_mask.shape[2] // fg_mask.shape[2],
                                       stride=fp_mask.shape[2] // fg_mask.shape[2]).bool()
        elif fp_mask.shape[2] < fg_mask.shape[2]:
            assert fg_mask.shape[2] % fp_mask.shape[2] == 0
            fp_mask = torch.repeat_interleave(fp_mask, repeats=fg_mask.shape[2]//fp_mask.shape[2], dim=2)
            fp_mask = torch.repeat_interleave(fp_mask, repeats=fg_mask.shape[3] // fp_mask.shape[3], dim=3)
        assert fp_mask.shape == fg_mask.shape
        fp_mask = torch.logical_and(fg_mask==0, fp_mask).detach().float()
        fp_scale_mask = torch.zeros_like(fp_mask)
        B, _, H, W = fg_mask.shape

        if self.distill_params['fp_scale_mode'] == 'average':
            for b in range(B):
                fp_scale_mask[b][fp_mask[b]>0] = 1.0 / torch.sum(fp_mask[b])
        elif self.distill_params['fp_scale_mode'] == 'dfs':

            for b in range(B):
                visited = torch.zeros_like(fg_mask[b][0]).bool()
                select_fp_mask = fp_mask[b][0]
                for coord in select_fp_mask.nonzero():
                    coord_y, coord_x = coord
                    if not visited[coord_y, coord_x]:
                        # dfs(coord_y, coord_x)
                        count = []
                        queue = []
                        queue.append(coord)
                        while len(queue) > 0:
                            current_coord = queue.pop(0)
                            current_coord_y, current_coord_x = current_coord
                            visited[current_coord_y, current_coord_x] = True
                            count.append(current_coord)
                            if (current_coord_y+1 < H and
                                    (not visited[current_coord_y+1, current_coord_x]) and
                                    select_fp_mask[current_coord_y+1, current_coord_x]):
                                queue.append((current_coord_y+1, current_coord_x))
                            if (current_coord_y -1 >=0 and
                                    (not visited[current_coord_y - 1, current_coord_x]) and
                                    select_fp_mask[current_coord_y - 1, current_coord_x]):
                                queue.append((current_coord_y-1, current_coord_x))
                            if (current_coord_x+1 < W and
                                    (not visited[current_coord_y, current_coord_x+1]) and
                                    select_fp_mask[current_coord_y, current_coord_x+1]):
                                queue.append((current_coord_y, current_coord_x+1))
                            if (current_coord_x-1 >= 0 and
                                    (not visited[current_coord_y, current_coord_x-1]) and
                                    select_fp_mask[current_coord_y, current_coord_x-1]):
                                queue.append((current_coord_y, current_coord_x-1))
                        num_points = len(count)
                        for coord_y, coord_x in count:
                            fp_scale_mask[b][0][coord_y, coord_x] = 1.0 / num_points

                        del count
                        del queue
                del visited
        else:
            raise NotImplementedError

        return fp_mask, fp_scale_mask, torch.sum(fp_mask, dim=(1,2,3))


    def fgd_distill_loss(self, teacher_feat, student_feat,
                         gt_bboxes_3d, gt_labels_3d,
                         canvas_feat,
                         heatmaps, teacher_preds, student_preds,
                         index):
        S_T = self.distill_params['spatial_t'] # 0.5
        s_ratio = self.distill_params['spatial_student_ratio'] # 1.0
        #   for channel attention
        C_T = self.distill_params['channel_t']  # 0.5
        # loss weight
        kd_fg_feat_loss_weight = self.distill_params['fg_feat_loss_weights'][index] \
            if len(self.distill_params['fg_feat_loss_weights']) > 1 else self.distill_params['fg_feat_loss_weights'][0]
        kd_bg_feat_loss_weight = self.distill_params['bg_feat_loss_weights'][index] \
            if len(self.distill_params['bg_feat_loss_weights']) > 1 else self.distill_params['bg_feat_loss_weights'][0]
        kd_channel_loss_weight = self.distill_params['channel_loss_weights'][index] \
            if len(self.distill_params['channel_loss_weights']) > 1 else self.distill_params['channel_loss_weights'][0]
        kd_spatial_loss_weight = self.distill_params['spatial_loss_weights'][index] \
            if len(self.distill_params['spatial_loss_weights']) > 1 else self.distill_params['spatial_loss_weights'][0]
        spatial_att = self.distill_params['spatial_attentions'][index] \
            if len(self.distill_params['spatial_attentions']) > 1 else self.distill_params['spatial_attentions'][0]
        feat_criterion = self.distill_params['feat_criterion']
        spatial_criterion = self.distill_params['spatial_criterion']
        channel_criterion = self.distill_params['channel_criterion']
        loss_dict = dict()
        feat_criterion = build_loss(feat_criterion)
        spatial_criterion = build_loss(spatial_criterion)
        channel_criterion = build_loss(channel_criterion)

        ##############
        # maybe a non-linear combination of spatial and channel adaptation would be the best
        teacher_feat = self.teacher_adaptations[index](teacher_feat)
        student_feat = self.channel_wise_adaptations[index](student_feat)
        ##############
        student_B, student_C, student_H, student_W = student_feat.size()
        teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
        assert student_B == teacher_B and student_H == teacher_H and student_W == teacher_W
        B = student_B

        # FIXME for now, only support nuscenes: LidarPoints/bboxes. Add support for other datasets in the future
        foreground_mask, fg_scale_mask, bg_scale_mask = self.foreground_scale_mask(student_H, student_W, gt_bboxes_3d,
                                                                                   self.distill_params['context_length'],
                                                                                   self.distill_params['context_weight'])
        foreground_mask, fg_scale_mask, bg_scale_mask = \
            foreground_mask.to(student_feat.device), fg_scale_mask.to(student_feat.device), bg_scale_mask.to(student_feat.device)
        foreground_mask, fg_scale_mask, bg_scale_mask = \
            foreground_mask.detach(), fg_scale_mask.detach(), bg_scale_mask.detach()
        if self.distill_params['foreground_mask'] in \
                ['gauss', 'extended_binary', 'gauss_plus_extended_binary', 'gauss_plus_extended_binary_clamp',
                 'gauss_in_gt', 'negative_linear_gauss_in_gt', 'gauss_plus_binary_clamp']:
            # For now, extended fg distill only support head to head distill
            assert self.distill_params['student_feat_pos'] == ['head'] and \
                   self.distill_params['teacher_feat_pos'] == ['head']
            if self.distill_params['custom_radius_func'] == 'centerpoint1':
                custom_radius_func = centerpoint_radius_func1
            elif self.distill_params['custom_radius_func'] == 'centerpoint2':
                custom_radius_func = centerpoint_radius_func2
            elif self.distill_params['custom_radius_func'] == 'centerpoint3':
                custom_radius_func = centerpoint_radius_func3
            elif self.distill_params['custom_radius_func'] == 'maxwh':
                custom_radius_func = partial(maxwh_radius_func, scale=self.distill_params['custom_radius_scale']) \
                    if hasattr(self.distill_params, 'custom_radius_scale') else maxwh_radius_func
            else:
                raise NotImplementedError

            # TODO: for each bbox, generate a center and a gaussian covariance matrix (pay attention to angle conversion)
            # get radius for each box (minimum covering length)
            # draw heatmap using gaussian_2d-like function
            # call draw_heatmap_gaussian. adapt it to 2d radius
            # call a mdraw scale function. adapt it to 2d radius


            custom_heatmaps, _, _, _, fg_scale_masks2 = multi_apply(
                self.pts_bbox_head.get_targets_single, gt_bboxes_3d, gt_labels_3d,
                custom_radius_func=custom_radius_func, custom_scale_func=draw_scale)
            custom_heatmaps = list(map(list, zip(*custom_heatmaps)))
            custom_heatmaps = [torch.stack(hms_) for hms_ in custom_heatmaps]
            custom_heatmaps = torch.cat(custom_heatmaps, dim=1)
            foreground_mask2 = torch.max(custom_heatmaps, dim=1, keepdim=True)[0]
            fg_scale_mask2 = [fg_scale_mask2.reshape(1,1,student_H,student_W) for fg_scale_mask2 in fg_scale_masks2]
            fg_scale_mask2 = torch.cat(fg_scale_mask2, dim=0)

            if self.distill_params['foreground_mask'] == 'extended_binary':
                foreground_mask = (foreground_mask2 != 0).float()
            elif self.distill_params['foreground_mask'] == 'gauss_plus_extended_binary':
                foreground_mask = foreground_mask2 + (foreground_mask2 != 0).float()
            elif self.distill_params['foreground_mask'] == 'gauss_plus_extended_binary_clamp':
                # FIXME this is a wrong implementation. It's no different from extended_binary'
                foreground_mask = foreground_mask2 + (foreground_mask2 != 0).float()
                foreground_mask = torch.clamp(foreground_mask, min=0.0, max=1.0)
            elif self.distill_params['foreground_mask'] == 'gauss_plus_binary_clamp':
                foreground_mask = self.distill_params['gauss_fg_weight'] * foreground_mask2 + (foreground_mask != 0).float()
                foreground_mask = torch.clamp(foreground_mask, min=0.0, max=1.0)
            elif self.distill_params['foreground_mask'] == 'gauss_in_gt':
                foreground_mask = (foreground_mask * foreground_mask2).detach()
            elif self.distill_params['foreground_mask'] == 'negative_linear_gauss_in_gt':
                foreground_mask2 = foreground_mask2 + kd_bg_feat_loss_weight / kd_fg_feat_loss_weight * (1 - foreground_mask2).float()
                foreground_mask = (foreground_mask * foreground_mask2).detach()
            else:
                foreground_mask = foreground_mask2.detach()
            # TODO: extend scale mask to gauss radius
        elif self.distill_params['foreground_mask'] != 'gt':
            raise NotImplementedError

        if getattr(self.distill_params, 'save_foreground_mask', False):
            for b in range(student_B):
                self.count += 1
                ndarr = foreground_mask[b].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                heatmap = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(self.distill_params['save_dir'], f'{self.count}th_foreground.jpg'),
                            heatmap)

        t_attention_mask = torch.mean(torch.abs(teacher_feat), [1], keepdim=True)
        t_attention_mask = t_attention_mask.view(B, -1)
        t_attention_mask = torch.softmax(t_attention_mask / S_T, dim=1) * teacher_H * teacher_W
        t_attention_mask = t_attention_mask.view(B, 1, teacher_H, teacher_W)

        s_attention_mask = torch.mean(torch.abs(student_feat), [1], keepdim=True)
        s_attention_mask = s_attention_mask.view(B, -1)
        s_attention_mask = torch.softmax(s_attention_mask / S_T, dim=1) * student_H * student_W
        s_attention_mask = s_attention_mask.view(B, 1, student_H, student_W)

        c_t_attention_mask = torch.mean(torch.abs(teacher_feat), [2, 3], keepdim=True)  # B x C x 1 x1
        c_t_attention_mask = c_t_attention_mask.view(B, -1)  # B x C
        c_t_attention_mask = torch.softmax(c_t_attention_mask / C_T, dim=1) * teacher_C
        c_t_attention_mask = c_t_attention_mask.view(B, teacher_C, 1, 1)  # B x C -> B x C x 1 x1

        if spatial_att == 'teacher':
            sum_attention_mask = t_attention_mask
            sum_attention_mask = sum_attention_mask.detach()
        elif spatial_att == 'teacher_student':
            sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
            sum_attention_mask = sum_attention_mask.detach()
        else:
            raise NotImplementedError
        c_sum_attention_mask = c_t_attention_mask
        c_sum_attention_mask = c_sum_attention_mask.detach()

        fg_mask = foreground_mask
        if self.distill_params['background_mask'] == 'logical_not':
            bg_mask = foreground_mask.logical_not()
        elif self.distill_params['background_mask'] == '1minus':
            bg_mask = 1 - foreground_mask
        else:
            raise NotImplementedError

        if self.distill_params['fp_as_foreground'][index] != 'none' and self._epoch >= self.distill_params['fp_epoch']:
            fp_mask, fp_scale_mask, fp_points_number = self.add_fp_as_fg(
                self.distill_params['fp_as_foreground'][index], foreground_mask,
                              heatmaps, teacher_preds, student_preds)
            bg_mask[fp_mask != 0] = 0
            bg_points_number = student_H * student_W - torch.sum(foreground_mask, dim=(1,2,3))
            for b in range(B):
                # in case an extremely weak model predicts all the points as positive
                if bg_points_number[b] > fp_points_number[b]:
                    bg_scale_mask[b][:] = 1.0 / (bg_points_number[b] - fp_points_number[b])
                else:
                    bg_scale_mask[b][:] = 0

        canvas_H, canvas_W = canvas_feat.shape[2:]
        assert canvas_H / student_H == canvas_H // student_H
        assert canvas_W / student_W == canvas_W // student_W
        factor_H = canvas_H // student_H
        factor_W = canvas_W // student_W
        non_empty_mask = F.max_pool2d(canvas_feat.max(dim=1, keepdim=True)[0], kernel_size=(factor_H, factor_W), stride=(factor_H, factor_W)).detach()
        non_empty_bg_mask = torch.logical_and(foreground_mask == 0, non_empty_mask != 0)
        # FIXME increase the weight of the points close to gt bboxes could lead to problems, as when foreground mask include those points
        # FIXME the weight of those points may become too high
        # increase the weight of non-empty background points
        if self.distill_params['non_empty_weight'] != 0:
            # for now, only support one position distill
            assert len(self.distill_params['student_feat_pos']) == 1 and \
                   len(self.distill_params['teacher_feat_pos']) == 1
            bg_mask[non_empty_bg_mask!=0] = 0
            assert foreground_mask.shape[1] == 1 and non_empty_bg_mask.shape[1] == 1
            # bg_points_number = student_H * student_W - torch.sum(foreground_mask!=0, dim=(1,2,3)) \
            #                    - torch.sum(non_empty_bg_mask, dim=(1,2,3))
            # bg_points_number = bg_points_number.reshape(B, 1, 1, 1).expand_as(bg_scale_mask)
            # bg_scale_mask = 1.0 / bg_points_number
            bg_points_number = student_H * student_W - torch.sum(foreground_mask, dim=(1, 2, 3))
            non_empty_points_number = torch.sum(non_empty_bg_mask, dim=(1,2,3))
            for b in range(B):
                bg_scale_mask[b][:] = 1.0 / (bg_points_number[b] - non_empty_points_number[b]) \
                    if self.distill_params['fp_as_foreground'][index] == 'none' \
                    else 1.0 / (bg_points_number[b] - non_empty_points_number[b] - fp_points_number[b])
            non_empty_bg_scale_mask = torch.zeros_like(bg_scale_mask)
            non_empty_bg_points_number = torch.sum(non_empty_bg_mask, dim=(1,2,3))
            for b in range(B):
                non_empty_bg_scale_mask[b][non_empty_bg_mask[b]] = 1.0 / non_empty_bg_points_number[b]
            non_empty_bg_mask = non_empty_bg_mask.float()

        if self.distill_params['scale_mask'] == 'combine_gt':
            # cannot assert for those wrong exps
            # assert self.distill_params['foreground_mask'] in ['gt', 'gauss_in_gt', 'negative_linear_gauss_in_gt']
            scale_mask = torch.maximum(fg_scale_mask, bg_scale_mask)
            fg_mask = fg_mask * scale_mask
            bg_mask = bg_mask * scale_mask
        elif self.distill_params['scale_mask'] == 'separate_gt':
            # cannot assert for those wrong exps
            # assert self.distill_params['foreground_mask'] in ['gt', 'gauss_in_gt', 'negative_linear_gauss_in_gt']
            fg_mask = fg_mask * fg_scale_mask
            bg_mask = bg_mask * bg_scale_mask
        elif self.distill_params['scale_mask'] == 'combine_extend':
            assert self.distill_params['foreground_mask'] in ['gauss', 'extended_binary', 'gauss_plus_extended_binary',
                                                              'gauss_plus_extended_binary_clamp', 'gauss_plus_binary_clamp']
            scale_mask = torch.maximum(fg_scale_masks2, bg_scale_mask)
            fg_mask = fg_mask * scale_mask
            bg_mask = bg_mask * scale_mask
        elif self.distill_params['scale_mask'] == 'separate_extend':
            assert self.distill_params['foreground_mask'] in ['gauss', 'extended_binary', 'gauss_plus_extended_binary',
                                                              'gauss_plus_extended_binary_clamp', 'gauss_plus_binary_clamp']
            fg_mask = fg_mask * fg_scale_mask2
            bg_mask = bg_mask * bg_scale_mask
        # for ablation
        elif self.distill_params['scale_mask'] == 'bg_only':
            fg_mask = fg_mask * bg_scale_mask
            bg_mask = bg_mask * bg_scale_mask
        elif self.distill_params['scale_mask']:
            raise NotImplementedError


        if getattr(self.distill_params, 'save_attention', False):

            import os
            filepath = './tools/visualization/masks_index1000_baseline/'
            os.makedirs(filepath, exist_ok=True)
            for b in range(student_B):


                self.count += 1
                # TODO: draw fg_mask & fp_mask in two colors. draw centerpoint to bevdepth4d attention mask. draw scale mask in heatmap
                palette = [[128, 128, 128], [0, 140, 255], [0, 0, 139],]
                # palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
                #            [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
                # TODO: test. DO I really need to multiply fg_mask with 255? it is already binary
                fg_mask_ndarr = foreground_mask[b,0,:,:].bool().to("cpu", torch.uint8).numpy()
                fp_mask_ndarr = fp_mask[b,0,:,:].bool().to("cpu", torch.uint8).numpy()
                fg_fp_mask_ndarr = fg_mask_ndarr + 2 * fp_mask_ndarr

                fg_fp_mask_img = Image.fromarray(fg_fp_mask_ndarr).convert('P')
                fg_fp_mask_img.putpalette(np.array(palette, dtype=np.uint8))
                # fg_fp_mask_img.save(f'./tools/visualization/masks/{b}th_{index}th_layer_fg_fp_mask_img.pdf')
                fg_fp_mask_img = fg_fp_mask_img.convert('RGB')
                fg_fp_mask_img.save(os.path.join(filepath, f'{b}th_{index}th_layer_fg_fp_mask_img.jpg'))

                # import pdb
                # pdb.set_trace()
                spatial_attention_mask_ndarr = sum_attention_mask[b,0,:,:] / (teacher_H * teacher_W)
                spatial_attention_mask_ndarr = spatial_attention_mask_ndarr.mul(250/torch.max(spatial_attention_mask_ndarr)).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
                spatial_attention_mask_ndarr = cv2.applyColorMap(spatial_attention_mask_ndarr, cv2.COLORMAP_VIRIDIS) #cv2.COLORMAP_PLASMA) #cv2.COLORMAP_INFERNO) # cv2.COLORMAP_MAGMA) #cv2.COLORMAP_CIVIDIS) # cv2.COLORMAP_JET)
                # spatial_attention_mask_img = Image.fromarray(spatial_attention_mask_ndarr).convert('P')
                # spatial_attention_mask_img.save(f'./tools/visualization/{index}th_layer_spatial_attention_mask_img.pdf')
                # cv2 imwrite doesn't support pdf
                cv2.imwrite(os.path.join(filepath, f'{b}th_{index}th_layer_spatial_attention_mask_img.jpg'), spatial_attention_mask_ndarr)

                t_attention_mask_ndarr = t_attention_mask[b,0,:,:] / (teacher_H * teacher_W)
                t_attention_mask_ndarr = t_attention_mask_ndarr.mul(250/torch.max(t_attention_mask_ndarr)).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
                t_attention_mask_ndarr = cv2.applyColorMap(t_attention_mask_ndarr, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(os.path.join(filepath, f'{b}th_{index}th_layer_teacher_spatial_attention_mask_img.jpg'),
                            t_attention_mask_ndarr)

                s_attention_mask_ndarr = s_attention_mask[b, 0, :, :] / (teacher_H * teacher_W)
                s_attention_mask_ndarr = s_attention_mask_ndarr.mul(250 / torch.max(s_attention_mask_ndarr)).add_(
                    0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
                s_attention_mask_ndarr = cv2.applyColorMap(s_attention_mask_ndarr, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(os.path.join(filepath, f'{b}th_{index}th_layer_student_spatial_attention_mask_img.jpg'),
                            s_attention_mask_ndarr)

                #TODO
                # scale mask as written in the paper
                total_scale_mask = torch.maximum(fg_scale_mask, bg_scale_mask)
                total_scale_mask = torch.maximum(total_scale_mask, fp_scale_mask)
                total_scale_mask_ndarr = total_scale_mask[b,0,:,:].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
                total_scale_mask_ndarr = cv2.applyColorMap(total_scale_mask_ndarr, cv2.COLORMAP_MAGMA) #cv2.COLORMAP_JET)
                # total_scale_mask_img = Image.fromarray(total_scale_mask_ndarr).convert('P')
                # total_scale_mask_img.save(f'./tools/visualization/{index}th_layer_total_scale_mask_img.pdf')
                # cv2 imwrite doesn't support pdf
                cv2.imwrite(os.path.join(filepath, f'{b}th_{index}th_layer_total_scale_mask_img.jpg'), total_scale_mask_ndarr)


        # TODO: add two scale_mask
        if self.distill_params['spatial_mask']:
            fg_mask = fg_mask * sum_attention_mask
            bg_mask = bg_mask * sum_attention_mask
        if self.distill_params['channel_mask']:
            fg_mask = fg_mask * c_sum_attention_mask
            bg_mask = bg_mask * c_sum_attention_mask
        kd_fg_feat_loss = (feat_criterion(student_feat, teacher_feat) * fg_mask).sum() \
                          * kd_fg_feat_loss_weight / B
        kd_bg_feat_loss = (feat_criterion(student_feat, teacher_feat) * bg_mask).sum() \
                          * kd_bg_feat_loss_weight / B

        # FIXME how to project student channel to teacher channel in a natural manner?
        loss_dict.update({'kd_fg_feat_loss': kd_fg_feat_loss})
        loss_dict.update({'kd_bg_feat_loss': kd_bg_feat_loss})
        if self.distill_params['channel_mask']:
            kd_channel_loss = channel_criterion(torch.mean(teacher_feat, [2, 3]),
                                                torch.mean(student_feat, [2, 3])).sum() \
                              * kd_channel_loss_weight / B
            loss_dict.update({'kd_channel_loss': kd_channel_loss})
        if self.distill_params['spatial_mask']:
            t_spatial_pool = torch.mean(teacher_feat, [1], keepdim=True).view(teacher_B, 1, teacher_H, teacher_W)
            s_spatial_pool = torch.mean(student_feat, [1], keepdim=True).view(student_B, 1, student_H, student_W)
            kd_spatial_loss = spatial_criterion(t_spatial_pool,
                                                self.spatial_wise_adaptations[index](s_spatial_pool)).sum() \
                              * kd_spatial_loss_weight / B
            loss_dict.update({'kd_spatial_loss': kd_spatial_loss})

        # FIXME when self.distill_params['fp_weight'] = kd_bg_feat_loss = 4e-2 using mode'average' and student fp, it seems that kd_fp_bg_feat_loss and kd_bg_feat_loss are on the same level
        # maybe student fp is not that important?
        if self.distill_params['fp_as_foreground'][index] != 'none' and self._epoch >= self.distill_params['fp_epoch']:
            # TODO: allow different fp weight for multi-scale distill
            fp_mask = fp_mask * fp_scale_mask * sum_attention_mask * c_sum_attention_mask
            kd_fp_bg_feat_loss = (feat_criterion(student_feat, teacher_feat) * fp_mask).sum() \
                                 * self.distill_params['fp_weight'] / B
            loss_dict.update({'kd_fp_bg_feat_loss': kd_fp_bg_feat_loss})

        if self.distill_params['non_empty_weight'] != 0:
            non_empty_bg_mask = non_empty_bg_mask * non_empty_bg_scale_mask * sum_attention_mask * c_sum_attention_mask
            kd_non_empty_bg_feat_loss = (feat_criterion(student_feat, teacher_feat) * non_empty_bg_mask).sum() \
                          * self.distill_params['non_empty_weight'] / B
            loss_dict.update({'kd_non_empty_bg_feat_loss': kd_non_empty_bg_feat_loss})

        if self.distill_params['affinity_mode'][index] == 'foreground':
            affinity_mask = foreground_mask != 0
        elif self.distill_params['affinity_mode'][index] == 'foreground+fp':
            assert self.distill_params['fp_as_foreground'][index] != 'none'
            affinity_mask = torch.logical_or(fp_mask != 0, foreground_mask != 0) \
                if self._epoch >= self.distill_params['fp_epoch'] else foreground_mask != 0
        # TODO: maybe visualization on attention selected region can help. But I am not sure if there is enough time
        elif self.distill_params['affinity_mode'][index] == 'attention':
            if hasattr(self.distill_params, 'affinity_attention_threshold'):
                affinity_mask = (sum_attention_mask / (teacher_H * teacher_W)) > self.distill_params['affinity_attention_threshold']
            else:
                topk_atts = torch.topk(sum_attention_mask.reshape(B, -1), k=self.distill_params['affinity_attention_topk'])
                topk_atts = topk_atts[0][:, -1]
                affinity_mask = torch.cat([mask.unsqueeze(0) > topk_att for mask, topk_att in zip(sum_attention_mask, topk_atts)], dim=0)

        elif self.distill_params['affinity_mode'][index] != 'none':
            raise NotImplementedError
        if self.distill_params['affinity_mode'][index] != 'none':
            # t_feat = [torch.cat([feat[c][mask[0, :, :]].unsqueeze(0) for c in range(teacher_C)], dim=0)
            #           for feat, mask in zip(teacher_feat, affinity_mask)]
            # s_feat = [torch.cat([feat[c][mask[0, :, :]].unsqueeze(0) for c in range(teacher_C)], dim=0)
            #           for feat, mask in zip(student_feat, affinity_mask)]
            t_feat = [torch.cat([feat[c][mask[0, :, :]].unsqueeze(-1) for c in range(teacher_C)], dim=-1)
                      for feat, mask in zip(teacher_feat, affinity_mask)]
            s_feat = [torch.cat([feat[c][mask[0, :, :]].unsqueeze(-1) for c in range(teacher_C)], dim=-1)
                      for feat, mask in zip(student_feat, affinity_mask)]
            loss_dict.update(self.affinity_distill_loss(t_feat, s_feat, index))


        return loss_dict


    def foreground_background_distill_loss(self, teacher_feat, student_feat, gt_bboxes_3d, index):
        student_B, student_C, student_H, student_W = student_feat.size()
        teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
        assert student_B == teacher_B and student_H == teacher_H and student_W == teacher_W
        B = student_B
        foreground_mask, scale_mask = self.foreground_scale_mask(student_H, student_W, gt_bboxes_3d)
        foreground_mask, scale_mask = foreground_mask.to(student_feat.device), scale_mask.to(student_feat.device)

        # loss weight
        kd_fg_feat_loss_weight = self.distill_params['fg_feat_loss_weights'][index] \
            if len(self.distill_params['fg_feat_loss_weights']) > 1 else self.distill_params['fg_feat_loss_weights'][0]
        kd_bg_feat_loss_weight = self.distill_params['bg_feat_loss_weights'][index] \
            if len(self.distill_params['bg_feat_loss_weights']) else self.distill_params['bg_feat_loss_weights'][0]
        fg_feat_criterion = self.distill_params['fg_feat_criterion']
        bg_feat_criterion = self.distill_params['bg_feat_criterion']
        fg_feat_criterion = build_loss(fg_feat_criterion)
        bg_feat_criterion = build_loss(bg_feat_criterion)
        scale = self.distill_params['scale_mask']
        loss_dict = dict()

        aligned_student_feat = self.adaptation_layers[index](student_feat)
        if scale:
            kd_fg_feat_loss = (fg_feat_criterion(aligned_student_feat, teacher_feat) * foreground_mask * scale_mask).sum() \
                              * kd_fg_feat_loss_weight / B
            kd_bg_feat_loss = (bg_feat_criterion(aligned_student_feat, teacher_feat) * foreground_mask.logical_not() * scale_mask).sum() \
                              * kd_bg_feat_loss_weight / B
        else:
            kd_fg_feat_loss = (fg_feat_criterion(aligned_student_feat, teacher_feat) * foreground_mask).sum() \
                              * kd_fg_feat_loss_weight / B
            kd_bg_feat_loss = (bg_feat_criterion(aligned_student_feat, teacher_feat) * foreground_mask.logical_not()).sum() \
                              * kd_bg_feat_loss_weight / B

        loss_dict.update({'kd_fg_feat_loss': kd_fg_feat_loss})
        loss_dict.update({'kd_bg_feat_loss': kd_bg_feat_loss})

        return loss_dict


    @force_fp32(apply_to=('teacher_feat', 'student_feat', 'teacher_preds', 'student_preds', 'heatmaps'))
    def distill_loss(self, teacher_feat, student_feat, teacher_preds, student_preds,
                     heatmaps, anno_boxes, inds, masks, gt_bboxes_3d, gt_labels_3d,
                     canvas_feat, index):
        # FIXME for now, only support tensor distillation.
        # TODO: add list distillation in the future
        # for input of size (256.704)
        # typically bevdet feature is of size (256,128,128)
        # centerpoint feature is of size (384,128,128)
        assert isinstance(teacher_feat, torch.Tensor) and isinstance(student_feat, torch.Tensor)
        # ensure each pixel on teacher feature map and student feature map have the same field-of-view
        if self.distill_type == 'all':
            losses_distill = self.all_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'foreground_background':
            losses_distill = self.foreground_background_distill_loss(teacher_feat, student_feat, gt_bboxes_3d, index)
        elif self.distill_type == 's2m2_ssd_heatmap':
            tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
            losses_distill = self.s2m2_ssd_heatmap_distill_loss(teacher_preds, student_preds, tp_mask, fp_mask, fn_mask)
        # elif self.distill_type == 's2m2_ssd_reg':
        #     tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
        elif self.distill_type == 's2m2_ssd_feature':
            tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
            losses_distill = self.s2m2_ssd_feature_distill_loss(teacher_feat, student_feat, tp_mask, fp_mask, fn_mask, index)
        elif self.distill_type == 'gauss_focal_heatmap':
            losses_distill = self.gauss_focal_heatmap_distill_loss(teacher_preds, student_preds, heatmaps)
        elif self.distill_type == 'non_local':
            losses_distill = self.non_local_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'affinity':
            losses_distill = self.affinity_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'linfengzhang':
            losses_distill = self.linfengzhang_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'fgd':
            losses_distill = self.fgd_distill_loss(teacher_feat, student_feat,
                                                   gt_bboxes_3d, gt_labels_3d,
                                                   canvas_feat,
                                                   heatmaps, teacher_preds, student_preds, index)
        else:
            raise NotImplementedError

        # if 'two_stage_epoch' in self.distill_params and self.distill_params['two_stage_epoch'] > 0:
        #     if self._epoch < self.distill_params['two_stage_epoch']:
        #         for key in losses_distill.keys():
        #             losses_distill[key] = 0 * losses_distill[key]

        return losses_distill

    def forward_distill(self, points, img_metas, gt_bboxes_3d, gt_labels_3d,
                        img_feats, lss_feat, bev_backbone_feats, preds, heatmaps,
                        img_inputs=None):
        # set `fp16_enabled` flag
        if hasattr(self, 'fp16_enabled') and self.fp16_enabled:
            for m in self.teacher_model.modules():
                if hasattr(m, 'fp16_enabled'):
                    m.fp16_enabled = True
        with torch.no_grad():
            if isinstance(self.teacher_model, (BEVDetSequentialES, BEVDepth4D)):
                assert isinstance(self.teacher_model, BEVDepth4D) # not implemented yet
                # img_feats, lss_feat, bev_backbone_feats
                (teacher_neck_feat, canvas_feat, teacher_backbone_feats), depth = self.teacher_model.extract_img_feat(
                    img_inputs, img_metas, return_lss_feature=True, return_backbone_feature=True)
                if getattr(self.distill_params, 'show_teacher_loss', False):
                    depth_gt = img_inputs[-1]
                    B, N, H, W = depth_gt.shape
                    depth_gt = torch.split(depth_gt.view(B, 2, N // 2, H, W), 1, 1)[0].squeeze(1)
                    loss_depth = self.teacher_model.get_depth_loss(depth_gt, depth)
                    losses_pts = self.teacher_model.forward_pts_train([teacher_neck_feat], gt_bboxes_3d,
                                                        gt_labels_3d, img_metas) # no gt_bboxes_ignore
                    logger = get_root_logger()
                    logger.info(f'teacher loss_depth: {loss_depth}')
                    # logger.info(f'teacher loss_pts: {losses_pts}')
                    logger.info(f'teacher loss sum: {loss_depth + sum(losses_pts.values())}')
            elif isinstance(self.teacher_model, CenterPoint):
                teacher_neck_feat, canvas_feat, teacher_backbone_feats = self.teacher_model.extract_pts_feat(
                    points, img_feats=None, img_metas=img_metas, return_canvas=True, return_backbone_feature=True)
            else:
                raise NotImplementedError
            assert teacher_neck_feat is not None and teacher_backbone_feats is not None
            if not isinstance(teacher_neck_feat, list):
                teacher_neck_feat = [teacher_neck_feat]
            teacher_preds = self.teacher_model.pts_bbox_head(teacher_neck_feat)
        assert sum([feats.requires_grad == False for feats in teacher_neck_feat]) / len(teacher_neck_feat) == 1
        if self.distill_type == 'fgd':
            assert isinstance(gt_bboxes_3d[0], LiDARInstance3DBoxes)
        # strongly correlate with specific model
        # FIXME pay 200% attention when coding!

        new_losses_distill = dict()
        assert len(list(set(self.distill_params['student_feat_pos']))) == len(self.distill_params['student_feat_pos'])
        assert len(list(set(self.distill_params['teacher_feat_pos']))) == len(self.distill_params['teacher_feat_pos'])
        assert len(self.distill_params['student_feat_pos']) == len(self.distill_params['teacher_feat_pos'])
        for index, (student_feat_pos, teacher_feat_pos) in \
                enumerate(zip(self.distill_params['student_feat_pos'], self.distill_params['teacher_feat_pos'])):
            if student_feat_pos == 'head':
                student_feat = img_feats[0]
            elif student_feat_pos == 'lss':
                student_feat = lss_feat
            elif student_feat_pos.startswith('backbone'):
                ##################
                # too early backbone feature distillation is bad for bevdet4d
                if self._epoch < self.distill_params['multi_scale_epoch']:
                    continue
                layer_index = int(student_feat_pos[-1])
                if isinstance(self.teacher_model, CenterPoint):
                    assert layer_index in range(3) # use teacher_adaptation to downsample
                else:
                    raise NotImplementedError
                student_feat = bev_backbone_feats[layer_index]
            else:
                raise NotImplementedError


            if teacher_feat_pos == 'head':
                teacher_feat = teacher_neck_feat[0]
            elif teacher_feat_pos.startswith('backbone'):
                layer_index = int(teacher_feat_pos[-1])
                if isinstance(self.teacher_model, CenterPoint):
                    assert layer_index in range(3)  # use teacher_adaptation to downsample
                else:
                    raise NotImplementedError
                teacher_feat = teacher_backbone_feats[layer_index]
            elif teacher_feat_pos == 'canvas':
                teacher_feat = canvas_feat
            else:
                raise NotImplementedError

            assert teacher_feat.shape[0] == student_feat.shape[0]
            assert student_feat.shape[2] / self.channel_wise_adaptations[index].stride[0] == \
                   teacher_feat.shape[2] / self.teacher_adaptations[index].stride[0]
            assert student_feat.shape[3] / self.channel_wise_adaptations[index].stride[1] == \
                   teacher_feat.shape[3] / self.teacher_adaptations[index].stride[1]

            losses_distill = self.distill_loss(teacher_feat=teacher_feat, student_feat=student_feat,
                                               teacher_preds=teacher_preds, student_preds=preds,
                                               heatmaps=heatmaps, anno_boxes=None, inds=None, masks=None,
                                               gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d,
                                               canvas_feat=canvas_feat.detach(), index=index)


            for key in losses_distill.keys():
                val = losses_distill[key]
                key = key + f'_{student_feat_pos}_{teacher_feat_pos}'
                new_losses_distill[key] = val
        return new_losses_distill


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        ############
        # if self.distill_type == 's2m2_ssd_feature' or (self.distill_type == 'fgd' and self.distill_params.fp_as_foreground=='teacher'):
        if self.distill_type == 's2m2_ssd_feature' or \
                (self.distill_type == 'fgd' and self.distill_params.fp_as_foreground=='teacher_selected_student') or \
                getattr(self.distill_params, 'clear_memory', False):
            self.count +=1
            if self.count == self.count_thres:
                torch.cuda.empty_cache()
                self.count = 0
        ############
        img_feats, lss_feat, bev_backbone_feats = self.extract_img_feat(img_inputs, img_metas, return_lss_feature=True, return_backbone_feature=True)
        img_feats = [img_feats]
        assert self.with_pts_bbox
        preds, (losses_pts, heatmaps, anno_boxes, inds, masks) = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, get_preds=True, get_targets=True)

        ##################
        if 'two_stage_epoch' in self.distill_params and self.distill_params['two_stage_epoch'] > 0:
            if self._epoch < self.distill_params['two_stage_epoch']:
                for key in losses_pts.keys():
                    losses_pts[key] = 0 * losses_pts[key]
        ##################
        losses = dict()
        losses.update(losses_pts)

        losses_distill = self.forward_distill(points, img_metas, gt_bboxes_3d, gt_labels_3d, img_feats, lss_feat, bev_backbone_feats, preds, heatmaps)
        losses.update(losses_distill)
        if getattr(self.distill_params, 'save_attention', False):
            import sys
            sys.exit()
        #########################
        return losses

    def to(self, *args, **kwargs):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model = self.teacher_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    # def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
    #     """
    #     Args:
    #         points (list[torch.Tensor]): the outer list indicates test-time
    #             augmentations and inner torch.Tensor should have a shape NxC,
    #             which contains all points in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch
    #         img (list[torch.Tensor], optional): the outer
    #             list indicates test-time augmentations and inner
    #             torch.Tensor should have a shape NxCxHxW, which contains
    #             all images in the batch. Defaults to None.
    #     """
    #     for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
    #         if not isinstance(var, list):
    #             raise TypeError('{} must be a list, but got {}'.format(
    #                 name, type(var)))
    #
    #     num_augs = len(img_inputs)
    #     if num_augs != len(img_metas):
    #         raise ValueError(
    #             'num of augmentations ({}) != num of image meta ({})'.format(
    #                 len(img_inputs), len(img_metas)))
    #
    #     if not isinstance(img_inputs[0][0],list):
    #         img_inputs = [img_inputs] if img_inputs is None else img_inputs
    #         points = [points] if points is None else points
    #         return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
    #     else:
    #         return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)
    #
    # def aug_test(self, points, img_metas, img=None, rescale=False):
    #     """Test function without augmentaiton."""
    #     combine_type = self.test_cfg.get('combine_type','output')
    #     if combine_type=='output':
    #         return self.aug_test_combine_output(points, img_metas, img, rescale)
    #     elif combine_type=='feature':
    #         return self.aug_test_combine_feature(points, img_metas, img, rescale)
    #     else:
    #         assert False
    #
    def simple_test(self, points, img_metas, img=None, rescale=False,
                    save_attention=False, save_dir=None, gt_bboxes_3d=None, gt_labels_3d=None, ):
        """Test function without augmentaiton."""
        torch.cuda.synchronize()
        start_time = time.time()
        feats = self.extract_feat(points, img=img, img_metas=img_metas)
        if len(feats) == 3:
            img_feats, pts_feats, depth = feats
        elif len(feats) == 2:
            img_feats, _ = feats
        else:
            raise NotImplementedError
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        torch.cuda.synchronize()
        end_time = time.time()

        # TODO: add heatmap save
        #################
        if save_attention:
            S_T = 1

            # attention related
            assert save_dir is not None
            teacher_img_feats, teacher_pts_feats = self.teacher_model.extract_feat(
                points, img=img, img_metas=img_metas)
            student_feat = img_feats[0].clone()
            teacher_feat = teacher_pts_feats[0].clone()

            student_B, student_C, student_H, student_W = student_feat.size()
            teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
            assert student_B == teacher_B
            B = student_B

            t_attention_mask = torch.mean(torch.abs(teacher_feat), [1], keepdim=True)
            t_attention_mask = t_attention_mask.view(B, -1)
            t_attention_mask = torch.softmax(t_attention_mask / S_T, dim=1) * teacher_H * teacher_W
            t_attention_mask = t_attention_mask.view(teacher_B, 1, teacher_H, teacher_W)

            s_attention_mask = torch.mean(torch.abs(student_feat), [1], keepdim=True)
            s_attention_mask = s_attention_mask.view(B, -1)
            s_attention_mask = torch.softmax(s_attention_mask / S_T, dim=1) * student_H * student_W
            s_attention_mask = s_attention_mask.view(student_B, 1, student_H, student_W)

            # c_t_attention_mask = torch.mean(torch.abs(teacher_feat), [2, 3], keepdim=True)  # B x C x 1 x1
            # c_t_attention_mask = c_t_attention_mask.view(B, -1)  # B x C
            # c_t_attention_mask = torch.softmax(c_t_attention_mask / C_T, dim=1) * teacher_C
            # c_t_attention_mask = c_t_attention_mask.view(teacher_B, teacher_C, 1, 1)  # B x C -> B x C x 1 x1
            #
            # c_s_attention_mask = torch.mean(torch.abs(student_feat), [2, 3], keepdim=True)
            # c_s_attention_mask = c_s_attention_mask.view(B, -1)
            # c_s_attention_mask = torch.softmax(c_s_attention_mask / C_T, dim=1) * student_C
            # c_s_attention_mask = c_s_attention_mask.view(student_B, student_C, 1, 1)

            # heatmap related
            student_preds = self.pts_bbox_head(img_feats)
            teacher_preds = self.teacher_model.pts_bbox_head(teacher_pts_feats)
            student_batch_hms = [preds_dict[0]['heatmap'].sigmoid() for preds_dict in student_preds]
            teacher_batch_hms = [preds_dict[0]['heatmap'].sigmoid() for preds_dict in teacher_preds]
            gt_batch_hms, _, _, _ = self.pts_bbox_head.get_targets(gt_bboxes_3d, gt_labels_3d)
            student_batch_hm = torch.cat(student_batch_hms, dim=1)
            teacher_batch_hm = torch.cat(teacher_batch_hms, dim=1)
            gt_batch_hm = torch.cat(gt_batch_hms, dim=1)
            student_batch_hm_max = torch.max(student_batch_hm, dim=1, keepdim=True)[0]
            teacher_batch_hm_max = torch.max(teacher_batch_hm, dim=1, keepdim=True)[0]
            gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]

            if self.distill_params['fp_as_foreground'][-1] != 'none':
                fp_mask, _, _ = self.add_fp_as_fg(torch.zeros_like(img_feats[0]), gt_batch_hms, teacher_preds, student_preds)

            if not hasattr(self, 'test_save_count'):
                self.test_save_count = 0

            from torchvision.utils import make_grid
            from PIL import Image
            import cv2
            for b in range(B):
                # grid = make_grid([t_attention_mask[b], s_attention_mask[b]])
                # # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
                # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                # heatmap = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(save_dir, f'{self.test_save_count}th_spatial_attention_st{S_T}.jpg'), heatmap)

                #
                # for task_id in range(len(student_batch_hms)):
                #     for cls_id in range(gt_batch_hms[task_id][b].shape[0]):
                #         grid = make_grid([gt_batch_hms[task_id][b][cls_id:cls_id+1,:,:],
                #                           teacher_batch_hms[task_id][b][cls_id:cls_id+1,:,:],
                #                           student_batch_hms[task_id][b][cls_id:cls_id+1,:,:]])
                #         ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                #         heatmap = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
                #         cv2.imwrite(os.path.join(save_dir, f'{self.test_save_count}th_{task_id}thtask_{cls_id}thclass_heatmap.jpg'), heatmap)
                #
                # for thres in [0.1, 0.3, 0.5]:
                #     tp_mask = torch.logical_and(gt_batch_hm_max > thres, student_batch_hm_max > thres) * 0.7
                #     fp_mask = torch.logical_and(gt_batch_hm_max < thres, student_batch_hm_max > thres) * 0.7
                #     fn_mask = torch.logical_and(gt_batch_hm_max > thres, student_batch_hm_max < thres) * 0.7
                #     grid = make_grid([gt_batch_hm_max[b], teacher_batch_hm_max[b], student_batch_hm_max[b], tp_mask[b], fp_mask[b], fn_mask[b]])
                #     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                #     heatmap = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
                #     cv2.imwrite(os.path.join(save_dir, f'{self.test_save_count}th_heatmap_max_tpfpfn_threshold{thres}.jpg'), heatmap)


                # grid = make_grid([teacher_feat[b, c:c + 1, :, :] for c in range(teacher_C)])
                # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                # heatmap = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(save_dir, f'{self.test_save_count}th_teacher_channel_featuremap.jpg'), heatmap)
                #
                # grid = make_grid([student_feat[b, c:c + 1, :, :] for c in range(student_C)])
                # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                # heatmap = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(save_dir, f'{self.test_save_count}th_teacher_channel_featuremap.jpg'), heatmap)

                if self.distill_params['fp_as_foreground'][-1] != 'none':
                    cv2.imwrite(os.path.join(save_dir, f'{self.test_save_count}th_fp_mask.jpg'), fp_mask[b][0].unsqueeze(2).cpu().numpy())

                self.test_save_count += 1
                # from mmcv.runner import get_dist_info
                # rank, world_size = get_dist_info()
                # if rank == 0:
                #     import pdb
                #     pdb.set_trace()
                # import torch.distributed as dist
                # dist.barrier()

        return bbox_list
    #
    #
    # def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
    #     img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
    #     from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
    #     img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
    #     bbox_list = [dict() for _ in range(1)]
    #     assert self.with_pts_bbox
    #     bbox_pts = self.simple_test_pts(
    #         img_feats, img_metas, rescale=False)
    #     for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
    #         result_dict['pts_bbox'] = pts_bbox
    #     return bbox_list
