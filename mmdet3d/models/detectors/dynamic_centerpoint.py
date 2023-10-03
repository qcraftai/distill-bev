# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F
from mmdet3d.core.points import LiDARPoints

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .. import builder
from mmdet3d.ops import Voxelization

@DETECTORS.register_module()
class DynamicCenterPoint(CenterPoint):
    """Use Dynamic Voxelization."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 test_dist2velo=False,
                 lidar_interval=1. / 20):
        super(DynamicCenterPoint,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg,
                             test_dist2velo, lidar_interval)

    def extract_pts_feat(self, pts, img_feats, img_metas, return_canvas=False, return_backbone_feature=False):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        outputs = []
        voxels, coors = self.voxelize(pts)
        coors = coors.type(torch.int32)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
        batch_size = int(coors[-1, 0].item() + 1)
        feature_coors = feature_coors.type(torch.int32)
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        if return_canvas:
            outputs.append(x)
        # x is a tuple of size (64, 256, 256)+(128, 128, 128)+(256, 64, 64) when x is of size (64, 512, 512)
        x = self.pts_backbone(x)
        if return_backbone_feature:
            outputs.append(x)
        # x is of size (384, 128, 128) when x is a tuple of size (64, 256, 256)+(128, 128, 128)+(256, 64, 64)
        if self.with_pts_neck:
            x = self.pts_neck(x)
            if len(outputs) == 0:
                outputs = x
            else:
                outputs.insert(0, x)
        return outputs


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch




@DETECTORS.register_module()
class DynamicMultiBranchCenterPoint(DynamicCenterPoint):
    """Do feature-level point cloud fusion. Use Dynamic Voxelization"""

    def __init__(self, pre_process=None, repeat=1, fuse='cat', max_multi_sweeps=None, time_thres=None,
                 pts_voxel_layer=None, pts_voxel_encoder=None, pts_middle_encoder=None,
                 **kwargs):
        super(DynamicMultiBranchCenterPoint, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, **kwargs)
        self.pre_process = pre_process is not None
        self.repeat = repeat
        assert repeat == 1 # only support 1 sweeps for now
        self.fuse = fuse
        self.max_multi_sweeps = max_multi_sweeps
        assert self.max_multi_sweeps is not None
        self.time_thres = time_thres
        assert self.time_thres is not None
        if self.pre_process:
            self.pre_process_nets = nn.ModuleList([builder.build_backbone(pre_process) for _ in range(self.repeat + 1)])
        # TODO build self.pts_voxel_encoder repeats

        self.pts_voxel_layers = nn.ModuleList([self.pts_voxel_layer] +
                                              [Voxelization(**pts_voxel_layer) for _ in range(repeat)])
        self.pts_voxel_encoders = nn.ModuleList([self.pts_voxel_encoder] +
                                                [builder.build_voxel_encoder(pts_voxel_encoder) for _ in range(repeat)])
        self.pts_middle_encoders = nn.ModuleList([self.pts_middle_encoder] +
                                                 [builder.build_middle_encoder(pts_middle_encoder) for _ in range(repeat)])
        assert all([type(self.pts_voxel_layers[i]) == type(self.pts_voxel_layers[0])
                    for i in range(len(self.pts_voxel_layers))])
        assert all([type(self.pts_voxel_encoders[i]) == type(self.pts_voxel_encoders[0])
                    for i in range(len(self.pts_voxel_encoders))])
        assert all([type(self.pts_middle_encoders[i]) == type(self.pts_middle_encoders[0])
                    for i in range(len(self.pts_middle_encoders))])


    def extract_pts_feat(self, pts, img_feats, img_metas, return_canvas=False, return_backbone_feature=False):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        outputs = []

        assert all([pt.shape[1]==5 for pt in pts])
        # for now, only works with nuscenes
        # assume ts - sweep_ts
        xs_list = []
        # TODO: get time range for differnt sweeps. allow true multi sweeps
        # for pt in pts:
        #     timestamps = torch.unique(pt[:,4], sorted=True)
        #     assert len(timestamps) <= self.max_sweeps
        #     quantile = torch.linspace(0, 1, self.repeat+2)
        #     timestamps_quantiles = torch.quantile(timestamps, quantile)
        #     xs = []
        #     for low_timestamp, high_timestamp in zip(timestamps_quantiles[:-1], timestamps_quantiles[1:]):
        #         # assume quantile always lies between two data points
        #         indices = torch.logical_and(pt[:,4] >= low_timestamp, pt[:,4] <= high_timestamp)
        #         voxels, coors = self.voxelize([pt[indices],])
        #         voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
        #         x = self.pts_middle_encoder(voxel_features, feature_coors)
        #         xs.append(x)
        #     xs_list.append(xs)
        for pt in pts:
            timestamps = torch.unique(pt[:,4], sorted=True)
            timestamps = timestamps[timestamps >= self.time_thres]
            assert len(timestamps) <= self.max_multi_sweeps
            # sometimes a sample only has limited prior sweeps
            time_thres = torch.min(timestamps) if len(timestamps) > 0 else self.time_thres
            xs = []

            # TODO: check if repeat current sweeps would do the job
            indices = pt[:, 4] < time_thres
            voxels, coors = self.voxelize([pt[indices], ], 0)
            coors = coors.type(torch.int32)
            voxel_features, feature_coors = self.pts_voxel_encoders[0](voxels, coors)
            x = self.pts_middle_encoders[0](voxel_features, feature_coors)
            xs.append(x[0])
            if len(timestamps) > 0:
                repeat_indices = pt[:, 4] >= time_thres
                voxels, coors = self.voxelize([pt[repeat_indices], ], 1)
                coors = coors.type(torch.int32)
                voxel_features, feature_coors = self.pts_voxel_encoders[1](voxels, coors)
                x = self.pts_middle_encoders[1](voxel_features, feature_coors)
            # repeat current sample if no prior sweeps
            # cannot do it in loading becasue here time threshold is used
            xs.append(x[0])
            xs_list.append(xs)
        xs_list = list(map(list, zip(*xs_list)))
        xs_list = [torch.cat(xs, dim=0) for xs in xs_list]
        xs_list = [pre_process_net(xs)[0] for xs, pre_process_net in zip(xs_list, self.pre_process_nets)]

        if self.fuse == 'cat':
            x = torch.cat(xs_list, dim=1)
        elif self.fuse == 'add':
            x = sum(xs_list)
        elif self.fuse == 'avg':
            x = sum(xs_list) / len(xs_list)
        else:
            raise NotImplementedError

        if return_canvas:
            outputs.append(x)
        # x is a tuple of size (64, 256, 256)+(128, 128, 128)+(256, 64, 64) when x is of size (64, 512, 512)
        x = self.pts_backbone(x)
        if return_backbone_feature:
            outputs.append(x)
        # x is of size (384, 128, 128) when x is a tuple of size (64, 256, 256)+(128, 128, 128)+(256, 64, 64)
        if self.with_pts_neck:
            x = self.pts_neck(x)
            if len(outputs) == 0:
                outputs = x
            else:
                outputs.insert(0, x)
        return outputs

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, index):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layers[index](res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

