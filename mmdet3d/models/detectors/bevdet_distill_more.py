# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn.functional as F

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS, ResNet
from .bevdet_distill import BEVDetDistill
from .. import builder

from ..necks import ViewTransformerLSSBEVDepthReproduce


@DETECTORS.register_module()
class BEVDet4DDistill(BEVDetDistill):
    def __init__(self, aligned=False, distill=None, pre_process=None,
                 pre_process_neck=None, detach=True, test_adj_ids=None,
                 before=False, interpolation_mode='bilinear', **kwargs):
        super(BEVDet4DDistill, self).__init__(**kwargs)
        # copied from BEVDetSequential
        self.aligned = aligned
        self.distill = distill is not None
        if self.distill:
            self.distill_net = builder.build_neck(distill)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = builder.build_neck(pre_process_neck)
        self.detach = detach
        self.test_adj_ids = test_adj_ids

        # copied from BEVDetSequentialES
        self.before = before
        self.interpolation_mode = interpolation_mode

    # copied from BEVDetSequentialES
    @force_fp32()
    def shift_feature(self, input, trans, rots):
        # here n means B and v means N in extract_img_feat
        n, c, h, w = input.shape
        _,v,_ =trans[0].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current lidar frame to adjacent lidar frame
        # transformation from current camera frame to current lidar frame
        c02l0 = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        c02l0[:,:,:3,:3] = rots[0]
        c02l0[:,:,:3,3] = trans[0]
        c02l0[:,:,3,3] = 1

        # transformation from adjacent camera frame to current lidar frame
        c12l0 = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        c12l0[:,:,:3,:3] = rots[1]
        c12l0[:,:,:3,3] = trans[1]
        c12l0[:,:,3,3] =1

        # lidaradj2lidarcurr is the same for all the cameras, so we only choose the first camera to compute it
        # transformation from current lidar frame to adjacent lidar frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:,0,:,:].view(n,1,1,4,4)
        '''
          c02l0 * inv（c12l0）
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        # remove z dim as we only align feature in bev plane
        l02l1 = l02l1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]

        # feat2bev 是特征空间和BEV空间（lidar坐标系）之间的变换，特征空间和lidar坐标系下的bev空间是不同的
        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0] # scaling
        feat2bev[1, 1] = self.img_view_transformer.dx[1] # scaling
        feat2bev[0, 2] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2. # translation
        feat2bev[1, 2] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2. # translation
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize, normalize是因为grid_sample要求要把绝对的坐标normalize到【-1,1】的区间内
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        # this operation shares the similar spirit with the inverse sampling in image rotation
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output

    # copied from BEVDetSequentialES
    # only change returned features
    def extract_img_feat(self, img, img_metas, return_lss_feature=False, return_backbone_feature=False):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        # FIXME note that here two frame fusion is hard cored in the code
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, _ , _, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)
            # bev_feat = self.img_view_transformer.voxel_pooling_accelerated(geom, volume)

            bev_feat_list.append(bev_feat)
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        # compatible with bevdet_distill return format
        # x = self.bev_encoder(bev_feat)
        # return [x]
        outputs = []
        if return_lss_feature:
            outputs.append(bev_feat)
        # (8, 64, 128, 128) -> (8, 256, 128, 128)
        x = self.img_bev_encoder_backbone(bev_feat)
        if return_backbone_feature:
            outputs.append(x)
        x = self.img_bev_encoder_neck(x)
        outputs.insert(0, x)

        return outputs




@DETECTORS.register_module()
class BEVDepthDistill(BEVDetDistill):
    def __init__(self, bevdepth_bev_forward=False, **kwargs):
        super(BEVDepthDistill, self).__init__(**kwargs)
        self.bevdepth_bev_forward = bevdepth_bev_forward
        if self.bevdepth_bev_forward:
            assert isinstance(self.img_bev_encoder_backbone, ResNet)

    # copied from BEVDepth_Base
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    # copied from BEVDepth_Base
    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        B, N, H, W = depth_gt.shape
        loss_weight = (~(depth_gt == 0)).reshape(B, N, 1, H, W).expand(B, N,
                                                                       self.img_view_transformer.D,
                                                                       H, W)
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                   /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                              self.img_view_transformer.D).to(torch.long)
        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32)
        depth = depth.sigmoid().view(B, N, self.img_view_transformer.D, H, W)

        loss_depth = F.binary_cross_entropy(depth, depth_gt_logit,
                                            weight=loss_weight)
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth

    def bev_encoder(self, x, return_backbone_feature=False):
        if not self.bevdepth_bev_forward:
            x = self.img_bev_encoder_backbone(x)
            trunk_outs = [None] + x
            x = self.img_bev_encoder_neck(x)
            # return x
        else:
            trunk_outs = [x]
            if self.img_bev_encoder_backbone.deep_stem:
                x = self.img_bev_encoder_backbone.stem(x)
            else:
                x = self.img_bev_encoder_backbone.conv1(x)
                x = self.img_bev_encoder_backbone.norm1(x)
                x = self.img_bev_encoder_backbone.relu(x)
            for i, layer_name in enumerate(self.img_bev_encoder_backbone.res_layers):
                res_layer = getattr(self.img_bev_encoder_backbone, layer_name)
                x = res_layer(x)
                if i in self.img_bev_encoder_backbone.out_indices:
                    trunk_outs.append(x)
            x = self.img_bev_encoder_neck(trunk_outs)
            ############
            if isinstance(x, list):
                assert len(x) == 1
                x = x[0]
            ############
        if return_backbone_feature:
            return x, trunk_outs[1:]
        else:
            return x


    # copied from BEVDepth_Base
    # only change returned features
    def extract_img_feat(self, img, img_metas, return_lss_feature=False, return_backbone_feature=False):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:])
        # x = self.bev_encoder(x)
        # return [x], depth
        outputs = []
        if return_lss_feature:
            outputs.append(x)
        # (8, 64, 128, 128) -> (8, 256, 128, 128)
        # x = self.img_bev_encoder_backbone(x)
        # if return_backbone_feature:
        #     outputs.append(x)
        # x = self.img_bev_encoder_neck(x)
        # outputs.insert(0, x)
        if return_backbone_feature:
            x, backbone_feature = self.bev_encoder(x, return_backbone_feature)
            outputs.append(backbone_feature)
            outputs.insert(0, x)
        else:
            x = self.bev_encoder(x)
            outputs.insert(0, x)

        return outputs, depth

    # merge forward_train in bevdet_distill and bevdepth. note extract_img_feat returns more outputs now
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
        if self.distill_type == 's2m2_ssd_feature' or \
                (self.distill_type == 'fgd' and self.distill_params.fp_as_foreground=='teacher_selected_student'):
            self.count +=1
            if self.count == self.count_thres:
                torch.cuda.empty_cache()
                self.count = 0
        ############
        (img_feats, lss_feat, bev_backbone_feats), depth = self.extract_img_feat(img_inputs, img_metas, return_lss_feature=True, return_backbone_feature=True)
        img_feats = [img_feats]
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        preds, (losses_pts, heatmaps, anno_boxes, inds, masks) = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                                                        gt_labels_3d, img_metas,
                                                                                        gt_bboxes_ignore,
                                                                                        get_preds=True,
                                                                                        get_targets=True)
        losses.update(losses_pts)
        if 'two_stage_epoch' in self.distill_params and self.distill_params['two_stage_epoch'] > 0:
            if self._epoch < self.distill_params['two_stage_epoch']:
                for key in losses_pts.keys():
                    losses_pts[key] = 0 * losses_pts[key]

        losses_distill = self.forward_distill(points, img_metas, gt_bboxes_3d, gt_labels_3d, img_feats, lss_feat, bev_backbone_feats, preds, heatmaps)
        losses.update(losses_distill)
        #########################
        return losses


# adopt __init__, shift_feature in BEVDet4DDistill; get_depth_loss in BEVDepthDistill
# copy extract_img_feat from BEVDepth4D and change return feature to be compatible with BEVDetDistill
@DETECTORS.register_module()
class BEVDepth4DDistill(BEVDet4DDistill, BEVDepthDistill):
    def __init__(self, bevdepth_bev_forward=False, **kwargs):
        super(BEVDepth4DDistill, self).__init__(**kwargs)
        self.bevdepth_bev_forward = bevdepth_bev_forward
        if self.bevdepth_bev_forward:
            assert isinstance(self.img_bev_encoder_backbone, ResNet)

    def bev_encoder(self, x, return_backbone_feature=False):
        if not self.bevdepth_bev_forward:
            x = self.img_bev_encoder_backbone(x)
            trunk_outs = [None] + x
            x = self.img_bev_encoder_neck(x)
        else:
            trunk_outs = [x]
            if self.img_bev_encoder_backbone.deep_stem:
                x = self.img_bev_encoder_backbone.stem(x)
            else:
                x = self.img_bev_encoder_backbone.conv1(x)
                x = self.img_bev_encoder_backbone.norm1(x)
                x = self.img_bev_encoder_backbone.relu(x)
            for i, layer_name in enumerate(self.img_bev_encoder_backbone.res_layers):
                res_layer = getattr(self.img_bev_encoder_backbone, layer_name)
                x = res_layer(x)
                if i in self.img_bev_encoder_backbone.out_indices:
                    trunk_outs.append(x)
            x = self.img_bev_encoder_neck(trunk_outs)
            ############
            if isinstance(x, list):
                assert len(x) == 1
                x = x[0]
            ############
        if return_backbone_feature:
            return x, trunk_outs[1:]
        else:
            return x

    def extract_img_feat(self, img, img_metas, return_lss_feature=False, return_backbone_feature=False):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            img_feat = self.img_view_transformer.featnet(x)
            depth_feat = x
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1)], dim=1)
            # TODO: bevdepth_view_transform
            depth_feat = self.img_view_transformer.se(depth_feat,
                                                      cam_params)
            depth_feat = self.img_view_transformer.extra_depthnet(depth_feat)[0]
            depth_feat = self.img_view_transformer.dcn(depth_feat)
            depth_digit = self.img_view_transformer.depthnet(depth_feat)
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans,
                                 self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        # x = self.bev_encoder(bev_feat)
        # return [x], depth_digit_list[0]
        outputs = []
        if return_lss_feature:
            outputs.append(bev_feat)
        if return_backbone_feature:
            x, backbone_feature = self.bev_encoder(bev_feat, return_backbone_feature)
            outputs.append(backbone_feature)
            outputs.insert(0, x)
        else:
            x = self.bev_encoder(bev_feat)
            outputs.insert(0, x)

        return outputs, depth_digit_list[0]

    # TODO: refactor forward_train code
    # merge forward_train in bevdet_distill and bevdepth4d. note extract_img_feat returns more outputs now
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
        if self.distill_type == 's2m2_ssd_feature' or \
                (self.distill_type == 'fgd' and self.distill_params.fp_as_foreground=='teacher_selected_student'):
            self.count +=1
            if self.count == self.count_thres:
                torch.cuda.empty_cache()
                self.count = 0
        ############
        (img_feats, lss_feat, bev_backbone_feats), depth = self.extract_img_feat(img_inputs, img_metas, return_lss_feature=True, return_backbone_feature=True)
        img_feats = [img_feats]
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        preds, (losses_pts, heatmaps, anno_boxes, inds, masks) = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                                                        gt_labels_3d, img_metas,
                                                                                        gt_bboxes_ignore,
                                                                                        get_preds=True,
                                                                                        get_targets=True)
        losses.update(losses_pts)
        if 'two_stage_epoch' in self.distill_params and self.distill_params['two_stage_epoch'] > 0:
            if self._epoch < self.distill_params['two_stage_epoch']:
                for key in losses_pts.keys():
                    losses_pts[key] = 0 * losses_pts[key]

        losses_distill = self.forward_distill(points, img_metas, gt_bboxes_3d, gt_labels_3d, img_feats, lss_feat, bev_backbone_feats, preds, heatmaps)
        losses.update(losses_distill)
        #########################
        return losses


@DETECTORS.register_module()
class BEVDepth4DReproduceOfficialDistill(BEVDepth4DDistill):
    def __init__(self, **kwargs):
        super(BEVDepth4DReproduceOfficialDistill, self).__init__(**kwargs)
        assert isinstance(self.img_view_transformer, ViewTransformerLSSBEVDepthReproduce)

    def extract_img_feat(self, img, img_metas, return_lss_feature=False, return_backbone_feature=False):
        # add sensor2ego
        inputs = img
        """Extract features of images."""
        # N for camera numbers. 6 in nuscenes
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, sensor2ego_rots, sensor2ego_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3),
                 sensor2ego_rots.view(B, 2, N, 3, 3),
                 sensor2ego_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans, sensor2ego_rots, sensor2ego_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran, sensor2ego_rot, sensor2ego_tran in zip(imgs, rots, trans,
                                                                      intrins, post_rots,
                                                                      post_trans,
                                                                      sensor2ego_rots, sensor2ego_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1),
                                   sensor2ego_rot.reshape(B * N, -1),
                                   sensor2ego_tran.reshape(B * N, -1),], dim=1)
            img_feat = x
            depth_feat = self.img_view_transformer._forward_depth_net(img_feat, cam_params)
            depth_digit = depth_feat[:, :self.img_view_transformer.D]
            depth_prob = self.img_view_transformer.get_depth_dist(depth_digit)
            img_feat = depth_feat[:, self.img_view_transformer.D:(self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = self.img_view_transformer._forward_voxel_net(volume)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        # x = self.bev_encoder(bev_feat)
        # return [x], depth_digit_list[0]
        outputs = []
        if return_lss_feature:
            outputs.append(bev_feat)
        if return_backbone_feature:
            x, backbone_feature = self.bev_encoder(bev_feat, return_backbone_feature)
            outputs.append(backbone_feature)
            outputs.insert(0, x)
        else:
            x = self.bev_encoder(bev_feat)
            outputs.insert(0, x)

        return outputs, depth_digit_list[0]



###############################
@DETECTORS.register_module()
class BEVDepth4DtoBEVDetDistill(BEVDetDistill):
    # 4D teacher model to bevdepth distillation
    def extract_img_feat(self, img, img_metas, return_lss_feature=False, return_backbone_feature=False):
        """Extract features of images."""
        # img contains image rots, trans, intrins, post_rots, post_trans, depth_gt
        # all in 4D format. ie, they are doubled.
        outputs = []

        inputs = img
        # N for camera numbers. 6 in nuscenes
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        imgs = imgs[0]

        # (8, 6, 3, 256, 704) -> (8, 6, 512, 16, 44)
        x = self.image_encoder(imgs)

        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3),]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        # extra = extra[0]
        extra = [t[0] for t in extra]
        # (8, 6, 512, 16, 44) -> (8, 64, 128, 128)
        x = self.img_view_transformer([x] + extra)
        if return_lss_feature:
            outputs.append(x)
        # (8, 64, 128, 128) -> (8, 256, 128, 128)
        x = self.img_bev_encoder_backbone(x)
        if return_backbone_feature:
            outputs.append(x)
        x = self.img_bev_encoder_neck(x)
        outputs.insert(0, x)
        return outputs

    # add img_inputs based on forwrad_train in bevdet_distill
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
        ############
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

        losses_distill = self.forward_distill(points, img_metas, gt_bboxes_3d, gt_labels_3d, img_feats, lss_feat, bev_backbone_feats, preds, heatmaps, img_inputs=img_inputs)
        losses.update(losses_distill)
        #########################
        return losses





@DETECTORS.register_module()
class BEVDepth4DtoBEVDepthDistill(BEVDepthDistill):
    # 4D teacher model to bevdepth distillation

    # based on extract_img_feat
    # change input preprocess, as it is 4D input
    def extract_img_feat(self, img, img_metas, return_lss_feature=False, return_backbone_feature=False):
        """Extract features of images."""
        # img contains image rots, trans, intrins, post_rots, post_trans, depth_gt
        # all in 4D format. ie, they are doubled.
        # x = self.image_encoder(img[0]) # original depth forward
        # x, depth = self.img_view_transformer([x] + img[1:])
        # BEVDepth4DtoBEVDepth forward
        inputs = img
        # N for camera numbers. 6 in nuscenes
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        imgs = imgs[0]

        x = self.image_encoder(imgs)

        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3),]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        # extra = extra[0]
        extra = [t[0] for t in extra] + [depth_gt]
        x, depth = self.img_view_transformer([x] + extra)
        # x = self.bev_encoder(x)
        # return [x], depth
        outputs = []
        if return_lss_feature:
            outputs.append(x)
        # (8, 64, 128, 128) -> (8, 256, 128, 128)
        # x = self.img_bev_encoder_backbone(x)
        # if return_backbone_feature:
        #     outputs.append(x)
        # x = self.img_bev_encoder_neck(x)
        # outputs.insert(0, x)
        if return_backbone_feature:
            x, backbone_feature = self.bev_encoder(x, return_backbone_feature)
            outputs.append(backbone_feature)
            outputs.insert(0, x)
        else:
            x = self.bev_encoder(x)
            outputs.insert(0, x)

        return outputs, depth

    # add split depth_gt code based on forward_train in BEVDepthDistill
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
        if self.distill_type == 's2m2_ssd_feature' or \
                (self.distill_type == 'fgd' and self.distill_params.fp_as_foreground=='teacher_selected_student'):
            self.count +=1
            if self.count == self.count_thres:
                torch.cuda.empty_cache()
                self.count = 0
        ############
        (img_feats, lss_feat, bev_backbone_feats), depth = self.extract_img_feat(img_inputs, img_metas, return_lss_feature=True, return_backbone_feature=True)
        img_feats = [img_feats]
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        preds, (losses_pts, heatmaps, anno_boxes, inds, masks) = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                                                        gt_labels_3d, img_metas,
                                                                                        gt_bboxes_ignore,
                                                                                        get_preds=True,
                                                                                        get_targets=True)
        losses.update(losses_pts)
        if 'two_stage_epoch' in self.distill_params and self.distill_params['two_stage_epoch'] > 0:
            if self._epoch < self.distill_params['two_stage_epoch']:
                for key in losses_pts.keys():
                    losses_pts[key] = 0 * losses_pts[key]

        losses_distill = self.forward_distill(points, img_metas, gt_bboxes_3d, gt_labels_3d, img_feats, lss_feat, bev_backbone_feats, preds, heatmaps, img_inputs=img_inputs)
        losses.update(losses_distill)
        #########################
        return losses