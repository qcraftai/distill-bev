import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np
from copy import deepcopy
from functools import partial

import mmcv
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint, load_state_dict, force_fp32
from mmdet.models import DETECTORS, build_detector
from mmdet.utils import get_root_logger
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import multi_apply
from mmdet3d.models.builder import build_loss
from mmdet3d.core.points import LiDARPoints
from mmdet3d.core.bbox import LiDARInstance3DBoxes, box_np_ops
from .bevformer import BEVFormer
from .lidarformer import LidarFormer
from .mvpformer import MVPFormer

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
import copy
from torch.utils.tensorboard import SummaryWriter

class TwoLayer(nn.Module):
    """
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

@DETECTORS.register_module()
class BEVFormerDistill(BEVFormer):
    def __init__(self, teacher_config, teacher_ckpt, distill_type, distill_params, eval_teacher=True, self_ckpt=None,
                 inherit_head=False, inherit_decoder=False, inherit_query=False, no_bg=False, img_norm_cfg=None, **kwargs):
        super(BEVFormerDistill, self).__init__(**kwargs)
        self.img_norm_cfg = img_norm_cfg
        self.eval_teacher = eval_teacher
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if isinstance(teacher_ckpt, str) and teacher_ckpt.lower() != 'none':
            print(f'loading teacher checkpoint from {teacher_ckpt}')
            # transpose teacher model
            ckpt = torch.load(teacher_ckpt, map_location=torch.device('cpu'))
            self.teacher_model.load_state_dict(ckpt['state_dict'], strict=True)

        if isinstance(self_ckpt, str) and self_ckpt.lower() != 'none':
            print(f'loading pretrained bevdet distill checkpoint')
            load_checkpoint(self, self_ckpt, map_location='cpu')

        self.inherit_head = inherit_head
        self.inherit_decoder = inherit_decoder
        self.inherit_query = inherit_query
        if self.inherit_head:
            assert isinstance(teacher_ckpt, str) and teacher_ckpt.lower() != 'none'
        if self.inherit_decoder:
            assert isinstance(teacher_ckpt, str) and teacher_ckpt.lower() != 'none'
        if self.inherit_query:
            assert isinstance(teacher_ckpt, str) and teacher_ckpt.lower() != 'none'

        self.distill_type = distill_type
        self.distill_params = distill_params
        assert distill_type in ['fgd',]
        assert 'heatmap' not in distill_type, 'to add heatmap distill support in the future!'

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
                                 stride=self.distill_params['student_adaptation_params']['downsample_stride'],
                                 padding=self.distill_params['student_adaptation_params']['downsample_padding'])
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
                            nn.Upsample(size=self.distill_params['student_adaptation_params']['upsample_out_size'],
                                          mode='bilinear', align_corners=True),
                            nn.Conv2d(student_channel, teacher_channel,
                                      kernel_size=1, stride=1, padding=0)
                        )
                    )
                elif adaptation_type == 'avgpool_1x1conv':
                    self.channel_wise_adaptations.append(
                        nn.Sequential(
                            nn.AvgPool2d(kernel_size=self.distill_params['student_adaptation_params']['downsample_kernel_size']),
                            nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0)
                        )
                    )
                    self.channel_wise_adaptations[index].stride = _pair(self.distill_params['student_adaptation_params']['downsample_kernel_size'])
                elif adaptation_type == 'interpolate_1x1conv':
                    self.channel_wise_adaptations.append(
                        nn.Sequential(
                            nn.Conv2d(student_channel, teacher_channel, kernel_size=1, stride=1, padding=0)
                        )
                    )
                else:
                    print(f'distill_params[adaptation_type]')
                    print(adaptation_type)
                    raise NotImplementedError

                if teacher_adaptation_type == 'avgpool':
                    self.teacher_adaptations.append(nn.AvgPool2d(**self.distill_params['teacher_adaptation_params']))
                    self.teacher_adaptations[index].stride = _pair(self.teacher_adaptations[index].stride)
                elif teacher_adaptation_type == 'maxpool':
                    self.teacher_adaptations.append(nn.MaxPool2d(**self.distill_params['teacher_adaptation_params']))
                    self.teacher_adaptations[index].stride = _pair(self.teacher_adaptations[index].stride)
                elif teacher_adaptation_type == 'identity':
                    self.teacher_adaptations.append(nn.Identity())
                    self.teacher_adaptations[index].stride = _pair(1)
                else:
                    raise NotImplementedError

            self.channel_wise_adaptations = nn.ModuleList(self.channel_wise_adaptations)
            self.teacher_adaptations = nn.ModuleList(self.teacher_adaptations)
            if distill_params['spatial_mask']:
                self.spatial_wise_adaptations = nn.ModuleList([
                    nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
                    for student_channel, teacher_channel in zip(student_channels, teacher_channels)])

        self.count = 0
        self.count_thres = 25

        self._epoch = 0

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('tensorboard path', os.getenv("TENSORBOARD_LOG_PATH", "/tensorboard_logs/"))
        self.writter = SummaryWriter(os.getenv("TENSORBOARD_LOG_PATH", "/tensorboard_logs/"))
        
        self.iter = 0

        self.no_bg = no_bg

    def init_weights(self):
        super(BEVFormerDistill, self).init_weights()
        self.inherit()

    def inherit(self):
        if self.inherit_head:
            logger = get_root_logger()
            logger.info(f'inherit teacher head')
            load_state_dict(self.pts_bbox_head.cls_branches, 
                            self.teacher_model.pts_bbox_head.cls_branches.state_dict(),
                            strict=False, logger=logger)
            load_state_dict(self.pts_bbox_head.reg_branches, 
                            self.teacher_model.pts_bbox_head.reg_branches.state_dict(),
                            strict=False, logger=logger)
            # load decoder weights
            if self.inherit_decoder:
                load_state_dict(self.pts_bbox_head.transformer.decoder, 
                                self.teacher_model.pts_bbox_head.transformer.decoder.state_dict(),
                                strict=False, logger=logger)
            # load object queries
            if self.inherit_query:
                load_state_dict(self.pts_bbox_head.query_embedding, 
                                self.teacher_model.pts_bbox_head.query_embedding.state_dict(),
                                strict=False, logger=logger)

    def set_epoch(self, epoch):
        self._epoch = epoch


    def non_local_distill_loss(self, teacher_feat, student_feat, index):
        weight = self.distill_params['nonlocal_weights'][index]
        criterion = self.distill_params['criterion']
        loss_dict = dict()

        if criterion.lower() == 'l1':
            criterion = partial(F.l1_loss, reduction='none')
        elif criterion.lower() == 'smoothl1':
            criterion = partial(F.smooth_l1_loss, reduction='none')
        elif criterion.lower() == 'mse':
            criterion = partial(F.mse_loss, reduction='none')
        else:
            raise NotImplementedError

        s_relation = self.student_non_locals[index](student_feat)
        t_relation = self.teacher_non_locals[index](teacher_feat)
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

    def query_distill_loss(self, teacher_feat, teacher_query, teacher_hs, student_feat, student_query, student_hs):
        teacher_feat = teacher_feat.reshape(teacher_feat.shape[0], teacher_feat.shape[1], -1).permute(0, 2, 1)
        student_feat = student_feat.reshape(student_feat.shape[0], student_feat.shape[1], -1).permute(0, 2, 1)
        teacher_query_sim = (teacher_feat@teacher_query[:, teacher_query.shape[1]//2:].T).sum(dim=-1)
        student_query_sim = (student_feat@student_query[:, student_query.shape[1]//2:].T).sum(dim=-1)
        teacher_hs_sim = torch.einsum('bij,bjkl->bikl', teacher_feat, teacher_hs.permute(1, 3, 0, 2)).sum(dim=-1)
        student_hs_sim = torch.einsum('bij,bjkl->bikl', student_feat, student_hs.permute(1, 3, 0, 2)).sum(dim=-1)
        weight = self.distill_params['query_loss_weight']
        criterion = build_loss(self.distill_params['query_criterion'])
        query_loss = criterion(teacher_query_sim, student_query_sim) +  criterion(teacher_hs_sim, student_hs_sim)
        return {'query_loss': query_loss * weight}

    def hs_distill_loss(self, teacher_feat, student_feat):
        loss_dict = dict()
        hs_feat_loss_weight = self.distill_params['hs_feat_loss_weights']
        feat_criterion = self.distill_params['feat_criterion']
        feat_criterion = build_loss(feat_criterion)
        student_B, student_C, _ = student_feat.size()
        hs_feat_loss = feat_criterion(student_feat, teacher_feat).sum() \
                          * hs_feat_loss_weight / student_B
        loss_dict.update({'hs_feat_loss': hs_feat_loss})
        return loss_dict

    def prob_cross_entropy(self, input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))


    def foreground_scale_mask(self, student_H, student_W, gt_bboxes_3d, bg_extend_length=0, bg_extend_weight=0,):
        grid_size = torch.tensor(self.pts_bbox_head.train_cfg['grid_size'])
        pc_range = torch.tensor(self.pts_bbox_head.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.pts_bbox_head.train_cfg['voxel_size'])
        assert grid_size[0] == grid_size[1] and student_W == student_H

        out_size_factor = grid_size[0] / student_W

        coord_xs = [i * voxel_size[0] * out_size_factor + pc_range[0] + voxel_size[0] * out_size_factor / 2 for i in range(student_W)]
        coord_ys = [i * voxel_size[1] * out_size_factor + pc_range[1] + voxel_size[1] * out_size_factor / 2 for i in range(student_H)]
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
        boxes_max = np.array([-np.inf, -np.inf, -np.inf])
        boxes_min = np.array([np.inf, np.inf, np.inf])
        for boxes in gt_bboxes_3d:
            points = feature_pixel_points.coord.numpy()
            boxes = deepcopy(boxes.tensor.numpy())
            for box in boxes:
                boxes_max = np.maximum(box[:3], boxes_max)
                boxes_min = np.minimum(box[:3], boxes_min)
            # the first three dimension marks the bottom center in LiDARInstance3DBoxes
            # unify z dim, 0 bottom center, 1 height
            boxes[:, 2] = 0
            boxes[:, 5] = 1
            mask = box_np_ops.points_in_rbbox(points, boxes) # NxM, N is the number of points (128x128), M is the number of bboxes

            foreground_mask = mask.any(axis=-1).astype(float)

            foreground_points_indices, bbox_indices = np.nonzero(mask)
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

        # print('boxes min', boxes_min, 'box max', boxes_max, 'coords min', np.min(coords, 0), 'coords max', np.max(coords, 0))

        return foreground_mask, fg_scale_mask, bg_scale_mask


    def add_fp_as_fg(self, mode, fg_mask, heatmaps, teacher_preds, student_preds):
        thres = self.distill_params['output_threshold']
        gt_thres = self.distill_params['groundtruth_threshold']
        if gt_thres is None:
            gt_thres = thres

        gt_batch_hm = [heatmap for heatmap in heatmaps]
        gt_batch_hm = torch.cat(gt_batch_hm, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]

        # FIXME check whether this is used and teacher hm outputs
        teacher_batch_hm = clip_sigmoid(teacher_preds)
        teacher_batch_hm_max = torch.max(teacher_batch_hm, dim=1, keepdim=True)[0]
        teacher_batch_hm_max = teacher_batch_hm_max.detach()

        # FIXME in centerpoint head loss function, student heatmaps has already been clip_sigomided
        # FIXME a easy mistake to make is to do sigmoid again
        # student_batch_hm = [clip_sigmoid(student_pred_dict[0]['heatmap']) for student_pred_dict in student_preds]
        # student_batch_hm = [student_pred_dict[0]['heatmap'] for student_pred_dict in student_preds]
        student_batch_hm = [student_pred_dict[0]['hm'] for student_pred_dict in student_preds]
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
        else:
            raise NotImplementedError

        return fp_mask, fp_scale_mask, torch.sum(fp_mask, dim=(1,2,3))

    
    def add_fp_as_fg_bbox(self, student_H, student_W, mode, fg_mask, teacher_preds, gt_bboxes_3d):
        thres = self.distill_params['output_threshold']
        gt_thres = self.distill_params['groundtruth_threshold']
        if gt_thres is None:
            gt_thres = thres

        grid_size = torch.tensor(self.pts_bbox_head.train_cfg['grid_size'])
        pc_range = torch.tensor(self.pts_bbox_head.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.pts_bbox_head.train_cfg['voxel_size'])
        # FIXME adpvatie out_size_factor. For now only support H=W
        # FIXME for now, only support nuscenes: LidarPoints/bboxes. Add support for other datasets in the future
        # out_size_factor = torch.tensor(self.pts_bbox_head.train_cfg['out_size_factor'])
        # FIXME out_size_factor calculation only supports integer divison
        assert grid_size[0] == grid_size[1] and student_W == student_H
        # assert grid_size[0] % student_W == 0
        # out_size_factor = grid_size[0] // student_W  
        out_size_factor = grid_size[0] / student_W

        coord_xs = [i * voxel_size[0] * out_size_factor + pc_range[0] + voxel_size[0] * out_size_factor / 2 for i in range(student_W)]
        coord_ys = [i * voxel_size[1] * out_size_factor + pc_range[1] + voxel_size[1] * out_size_factor / 2 for i in range(student_H)]
        coord_xs, coord_ys = np.meshgrid(coord_xs, coord_ys, indexing='ij')
        coord_xs = coord_xs.reshape(-1, 1)
        coord_ys = coord_ys.reshape(-1, 1)
        # make z dim 0.5
        coord_zs = np.ones_like(coord_xs) * 0.5
        coords = np.hstack((coord_xs, coord_ys, coord_zs))
        assert coords.shape[0] == student_W * student_W and coords.shape[1] == 3
        feature_pixel_points = LiDARPoints(torch.tensor(coords), 3, attribute_dims=None)
        points = feature_pixel_points.coord.numpy()

        fp_masks = []

        # step 1: acquire the bboxes of all predictions
        # step 2: acquire the bboxes of all gts
        # step 3: (Pred > thres) & (Gt < thres)
        for (bboxes, scores, labels), gt_bboxes in zip(teacher_preds, gt_bboxes_3d):
            # pre-filter for valid bboxes
            indices = (scores.cpu().numpy() > thres).nonzero()
            valid_bboxes = bboxes[indices]
            
            boxes = deepcopy(valid_bboxes.tensor.cpu().numpy())
            # the first three dimension marks the bottom center in LiDARInstance3DBoxes
            # unify z dim, 0 bottom center, 1 height
            boxes[:, 2] = 0
            boxes[:, 5] = 1
            mask = box_np_ops.points_in_rbbox(points, boxes) # NxM, N is the number of points (128x128), M is the number of bboxes

            fp_mask = torch.tensor(mask.any(axis=-1).astype(float)).to(scores.device)

            gt_bboxes = deepcopy(gt_bboxes.tensor.numpy())
            # the first three dimension marks the bottom center in LiDARInstance3DBoxes
            # unify z dim, 0 bottom center, 1 height
            gt_bboxes[:, 2] = 0
            gt_bboxes[:, 5] = 1
            gt_mask = box_np_ops.points_in_rbbox(points, gt_bboxes) # NxM, N is the number of points (128x128), M is the number of bboxes

            gt_mask = gt_mask.any(axis=-1).astype(float)

            re_gt_mask = torch.tensor(1 - gt_mask).to(scores.device)

            fp_mask =  torch.logical_and(fp_mask, re_gt_mask).detach().float()

            fp_masks.append(fp_mask.reshape(1, 1, student_H, student_W))

        fp_masks = torch.cat(fp_masks, dim=0)
        fp_scale_masks = torch.zeros_like(fp_masks)
        B = fp_masks.shape[0]

        if self.distill_params['fp_scale_mode'] == 'average':
            for b in range(B):
                fp_scale_masks[b][fp_masks[b]>0] = 1.0 / torch.sum(fp_masks[b])
        else:
            raise NotImplementedError

        return fp_masks, fp_scale_masks, torch.sum(fp_masks, dim=(1,2,3))


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
        kd_spatial_loss_weight = self.distill_params['spatial_loss_weights'][index] \
            if len(self.distill_params['spatial_loss_weights']) > 1 else self.distill_params['spatial_loss_weights'][0]
        spatial_att = self.distill_params['spatial_attentions'][index] \
            if len(self.distill_params['spatial_attentions']) > 1 else self.distill_params['spatial_attentions'][0]
        feat_criterion = self.distill_params['feat_criterion']
        spatial_criterion = self.distill_params['spatial_criterion']
        loss_dict = dict()
        feat_criterion = build_loss(feat_criterion)
        spatial_criterion = build_loss(spatial_criterion)

        ##############
        # maybe a non-linear combination of spatial and channel adaptation would be the best
        # print('teacher_feat.size()', teacher_feat.size())
        teacher_feat = self.teacher_adaptations[index](teacher_feat)
        # print('teacher_feat.size() after adaptation', teacher_feat.size())
        teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
        # print('student_feat.size()', student_feat.size())
        if self.distill_params['adaptation_type'][index] == 'interpolate_1x1conv':
            student_feat = nn.functional.interpolate(student_feat, (teacher_H, teacher_W), mode='bilinear', align_corners=True)
        student_feat = self.channel_wise_adaptations[index](student_feat)
        # print('student_feat.size() after adaptation', student_feat.size())

        ##############
        student_B, student_C, student_H, student_W = student_feat.size()
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
        if self.distill_params['foreground_mask'] != 'gt':
            raise NotImplementedError

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
            fp_mask, fp_scale_mask, fp_points_number = self.add_fp_as_fg_bbox(student_H, student_W,
                self.distill_params['fp_as_foreground'][index], foreground_mask,
                              teacher_preds, gt_bboxes_3d)
            bg_mask[fp_mask != 0] = 0
            bg_points_number = student_H * student_W - torch.sum(foreground_mask, dim=(1,2,3))
            for b in range(B):
                # in case an extremely weak model predicts all the points as positive
                if bg_points_number[b] > fp_points_number[b]:
                    bg_scale_mask[b][:] = 1.0 / (bg_points_number[b] - fp_points_number[b])
                else:
                    bg_scale_mask[b][:] = 0

        if self.distill_params['scale_mask'] == 'combine_gt':
            scale_mask = torch.maximum(fg_scale_mask, bg_scale_mask)
            fg_mask = fg_mask * scale_mask
            bg_mask = bg_mask * scale_mask
        elif self.distill_params['scale_mask'] == 'separate_gt':
            fg_mask = fg_mask * fg_scale_mask
            bg_mask = bg_mask * bg_scale_mask
        # for ablation
        elif self.distill_params['scale_mask'] == 'bg_only':
            fg_mask = fg_mask * bg_scale_mask
            bg_mask = bg_mask * bg_scale_mask
        elif self.distill_params['scale_mask']:
            raise NotImplementedError

        if self.distill_params['spatial_mask']:
            fg_mask = fg_mask * sum_attention_mask
            bg_mask = bg_mask * sum_attention_mask
        kd_fg_feat_loss = (feat_criterion(student_feat, teacher_feat) * fg_mask).sum() \
                          * kd_fg_feat_loss_weight / B
        kd_bg_feat_loss = (feat_criterion(student_feat, teacher_feat) * bg_mask).sum() \
                          * kd_bg_feat_loss_weight / B

        loss_dict.update({'kd_fg_feat_loss': kd_fg_feat_loss})
        if not self.no_bg:
            loss_dict.update({'kd_bg_feat_loss': kd_bg_feat_loss})
        if self.distill_params['spatial_mask']:
            t_spatial_pool = torch.mean(teacher_feat, [1], keepdim=True).view(teacher_B, 1, teacher_H, teacher_W)
            s_spatial_pool = torch.mean(student_feat, [1], keepdim=True).view(student_B, 1, student_H, student_W)
            kd_spatial_loss = spatial_criterion(t_spatial_pool,
                                                self.spatial_wise_adaptations[index](s_spatial_pool)).sum() \
                              * kd_spatial_loss_weight / B
            loss_dict.update({'kd_spatial_loss': kd_spatial_loss})

        if self.distill_params['fp_as_foreground'][index] != 'none' and self._epoch >= self.distill_params['fp_epoch']:
            fp_mask = fp_mask * fp_scale_mask * sum_attention_mask * c_sum_attention_mask
            kd_fp_bg_feat_loss = (feat_criterion(student_feat, teacher_feat) * fp_mask).sum() \
                                 * self.distill_params['fp_weight'] / B
            loss_dict.update({'kd_fp_bg_feat_loss': kd_fp_bg_feat_loss})

        if self.distill_params['affinity_mode'][index] == 'foreground':
            affinity_mask = foreground_mask != 0
        elif self.distill_params['affinity_mode'][index] == 'foreground+fp':
            assert self.distill_params['fp_as_foreground'][index] != 'none'
            affinity_mask = torch.logical_or(fp_mask != 0, foreground_mask != 0) \
                if self._epoch >= self.distill_params['fp_epoch'] else foreground_mask != 0
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
            t_feat = [torch.cat([feat[c][mask[0, :, :]].unsqueeze(-1) for c in range(teacher_C)], dim=-1)
                      for feat, mask in zip(teacher_feat, affinity_mask)]
            s_feat = [torch.cat([feat[c][mask[0, :, :]].unsqueeze(-1) for c in range(teacher_C)], dim=-1)
                      for feat, mask in zip(student_feat, affinity_mask)]
            loss_dict.update(self.affinity_distill_loss(t_feat, s_feat, index))


        return loss_dict


    @force_fp32(apply_to=('teacher_feat', 'student_feat', 'teacher_preds', 'student_preds', 'heatmaps'))
    def distill_loss(self, teacher_feat, student_feat, teacher_preds, student_preds,
                     teacher_query, student_query,
                     teacher_hs, student_hs,
                     heatmaps, anno_boxes, inds, masks, gt_bboxes_3d, gt_labels_3d,
                     canvas_feat, index):
        # for input of size (256.704)
        # typically bevdet feature is of size (256,128,128)
        # centerpoint feature is of size (384,128,128)
        assert isinstance(teacher_feat, torch.Tensor) and isinstance(student_feat, torch.Tensor)
        # ensure each pixel on teacher feature map and student feature map have the same field-of-view
        if self.distill_type == 'fgd':
            losses_distill = self.fgd_distill_loss(teacher_feat, student_feat,
                                                   gt_bboxes_3d, gt_labels_3d,
                                                   canvas_feat,
                                                   heatmaps, teacher_preds, student_preds, index)
        else:
            raise NotImplementedError

        # query based distillation loss
        if 'query_criterion' in self.distill_params and self.distill_params['query_criterion'] != 'none' and index==0:
            losses_distill.update(self.query_distill_loss(teacher_feat, teacher_query, teacher_hs, student_feat, student_query, student_hs))


        return losses_distill


    def forward_distill(self, points, img_metas, gt_bboxes_3d, gt_labels_3d,
                        img_feats, lss_feat, bev_backbone_feats, student_query, hs_feats, preds, heatmaps,
                        img_inputs=None):
        # set `fp16_enabled` flag
        if hasattr(self, 'fp16_enabled') and self.fp16_enabled:
            for m in self.teacher_model.modules():
                if hasattr(m, 'fp16_enabled'):
                    m.fp16_enabled = True
        with torch.no_grad():
            if isinstance(self.teacher_model, (LidarFormer, MVPFormer)):
                teacher_preds = []
                _, teacher_x = self.teacher_model.extract_feat(points, None, img_metas)
                teacher_outs = self.teacher_model.pts_bbox_head(teacher_x)
                teacher_prehead_feat = teacher_outs['bev_embed']
                teacher_hs_feat = teacher_outs['hs']
                teacher_query = teacher_outs['query_embed']
                teacher_preds = self.teacher_model.pts_bbox_head.get_bboxes(teacher_outs, img_metas, rescale=False)
            else:
                raise NotImplementedError

        if self.distill_type == 'fgd':
            assert isinstance(gt_bboxes_3d[0], LiDARInstance3DBoxes)

        new_losses_distill = dict()
        assert len(list(set(self.distill_params['student_feat_pos']))) == len(self.distill_params['student_feat_pos'])
        assert len(list(set(self.distill_params['teacher_feat_pos']))) == len(self.distill_params['teacher_feat_pos'])
        assert len(self.distill_params['student_feat_pos']) == len(self.distill_params['teacher_feat_pos'])
        for index, (student_feat_pos, teacher_feat_pos) in \
                enumerate(zip(self.distill_params['student_feat_pos'], self.distill_params['teacher_feat_pos'])):
            if student_feat_pos == 'head':
                student_feat = img_feats
            elif student_feat_pos == 'lss':
                student_feat = lss_feat
            elif student_feat_pos.startswith('backbone'):
                if self._epoch < self.distill_params['multi_scale_epoch']:
                    continue
                layer_index = int(student_feat_pos[-1])
                student_feat = bev_backbone_feats[layer_index]
            elif student_feat_pos == 'hs':
                student_feat = torch.squeeze(hs_feats)
            else:
                raise NotImplementedError


            if teacher_feat_pos == 'head':
                teacher_feat = teacher_prehead_feat
            elif teacher_feat_pos.startswith('backbone'):
                layer_index = int(teacher_feat_pos[-1])
                teacher_feat = teacher_backbone_feats[layer_index]
            elif teacher_feat_pos == 'hs':
                teacher_feat = torch.squeeze(teacher_hs_feat)
            elif teacher_feat_pos == 'canvas':
                teacher_feat = canvas_feat
            else:
                raise NotImplementedError

            if student_feat_pos != 'hs' and teacher_feat_pos != 'hs':
                student_feat = student_feat.permute(0, 2, 1)
                teacher_feat = teacher_feat.permute(0, 2, 1)

                sH = sW = int(student_feat.shape[2]**0.5)
                tH = tW = int(teacher_feat.shape[2]**0.5)
                student_feat = student_feat.reshape(student_feat.shape[0], student_feat.shape[1], sH, sW)
                teacher_feat = teacher_feat.reshape(teacher_feat.shape[0], teacher_feat.shape[1], tH, tW)
                
                assert teacher_feat.shape[0] == student_feat.shape[0]
            
                losses_distill = self.distill_loss(teacher_feat=teacher_feat, student_feat=student_feat,
                                                teacher_preds=teacher_preds, student_preds=preds,
                                                teacher_query=teacher_query, student_query=student_query,
                                                teacher_hs=teacher_hs_feat, student_hs=hs_feats,
                                                heatmaps=heatmaps, anno_boxes=None, inds=None, masks=None,
                                                gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d,
                                                canvas_feat=None, index=index)
            else:
                losses_distill = self.hs_distill_loss(teacher_feat, student_feat)


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
                      img=None,
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
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        outs, losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, get_preds=True)
        losses.update(losses_pts)

        preds, student_feats, student_query, student_hs = outs['all_bbox_preds'], outs['bev_embed'], outs['query_embed'], outs['hs']

        losses_distill = self.forward_distill(points, img_metas, gt_bboxes_3d, gt_labels_3d, student_feats, None, None, student_query, student_hs, preds, None)
        losses.update(losses_distill)
        for k in losses:
            self.writter.add_scalar(k, losses[k].detach().cpu().numpy(), self.iter)
        self.iter += 1
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

