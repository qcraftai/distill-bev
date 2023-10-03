import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS
from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32
from mmcv.ops import SyncBatchNorm
from mmcv.cnn import build_norm_layer
from mmdet.models.backbones.resnet import BasicBlock

from .view_transformer_mine import ViewTransformerLiftSplatShoot

class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """
    def __init__(self, in_channels, mid_channels, out_channels, norm_cfg):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            # nn.BatchNorm2d(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            # nn.BatchNorm2d(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            # nn.BatchNorm2d(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @force_fp32()
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 norm_cfg):
                 # BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        # self.bn = BatchNorm(planes)
        self.bn = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256,
                 norm_cfg=dict(type='BN')):
                 # BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 norm_cfg=norm_cfg)
                                 # BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 norm_cfg=norm_cfg)
                                # BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 norm_cfg=norm_cfg)
                                # BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 norm_cfg=norm_cfg)
                                 # BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            # BatchNorm(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        # self.bn1 = BatchNorm(mid_channels)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, norm_cfg):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # nn.BatchNorm2d(mid_channels),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        # self.bn = nn.BatchNorm1d(45) # FIXME change this magic number
        # import pdb
        # pdb.set_trace()
        # if 'syncbn' in norm_cfg.type.lower():
        #     self.bn = build_norm_layer(dict(type='naiveSyncBN0d'), 45)[1]
        # else:
        self.bn = build_norm_layer(norm_cfg, 45)[1]
        self.depth_mlp = Mlp(45, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(45, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            # build_conv_layer(cfg=dict(
            #     type='DCN',
            #     in_channels=mid_channels,
            #     out_channels=mid_channels,
            #     kernel_size=3,
            #     padding=1,
            #     groups=4,
            #     im2col_step=128,
            # )),
            build_conv_layer(dict(type='DCNv2',deform_groups=1),
                             in_channels=mid_channels,
                             out_channels=mid_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             dilation=1,
                             bias=False,
            ),
            # nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, dilation=1,),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, cam_params):
        mlp_input = self.bn(cam_params.reshape(-1, cam_params.shape[-1], 1, 1)).squeeze(dim=3).squeeze(dim=2)
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)

@NECKS.register_module()
class ViewTransformerLSSBEVDepthReproduce(ViewTransformerLiftSplatShoot):
    def __init__(self, depth_net_conf, loss_depth_weight, norm_cfg=dict(type='BN'), **kwargs):
        super(ViewTransformerLSSBEVDepthReproduce, self).__init__(**kwargs)
        del self.depthnet
        self.depth_net = self._configure_depth_net(depth_net_conf, norm_cfg)
        self.loss_depth_weight = loss_depth_weight
        # 'depth_net_conf':
        # dict(in_channels=512, mid_channels=512)

        self.depth_aggregation_net = DepthAggregation(self.numC_Trans, self.numC_Trans, self.numC_Trans, norm_cfg)

    def _configure_depth_net(self, depth_net_conf, norm_cfg):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.numC_Trans,
            self.D,
            norm_cfg,
        )

    def _forward_depth_net(self, feat, cam_params):
        return self.depth_net(feat, cam_params)

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.depth_aggregation_net(img_feat_with_depth).view(
                n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, sensor2egos, depth_gt = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        img_feat = x
        # official implementation will select non zero element here
        # TODO: add sensor to ego here
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1),
                               sensor2egos.reshape(B*N,-1),],dim=1)
        depth_feat = self._forward_depth_net(img_feat, cam_params)
        depth_digit = depth_feat[:, :self.D]
        depth_prob = self.get_depth_dist(depth_digit)
        img_feat = depth_feat[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = self._forward_voxel_net(volume)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # Splat
        # geom is of size (6, 59, 16, 44, 3)
        # volume is of size (8, 6, 59, 16, 44, 64), 64 is numC_Trans
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(geom, volume)
        else:
            bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat, depth_digit