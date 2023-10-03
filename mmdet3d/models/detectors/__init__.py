# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .centerpoint import CenterPoint
from .dynamic_centerpoint import DynamicCenterPoint
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .mvx_two_stage import MVXTwoStageDetector
from .bevdet import BEVDet, BEVDetSequential
from .bevdet_distill import BEVDetDistill
from .bevdet_distill_more import BEVDet4DDistill, BEVDepthDistill, BEVDepth4DDistill
from .lidarformer import LidarFormer
from .mvpformer import MVPFormer
from .bevformer import BEVFormer
from .bevformer_distill import BEVFormerDistill

__all__ = [
    'Base3DDetector', 'MVXTwoStageDetector', 'DynamicMVXFasterRCNN', 'MVXFasterRCNN',  
    'CenterPoint', 'DynamicCenterPoint', 'BEVDet', 'BEVDetSequential', 'BEVDetDistill',
    'BEVDet4DDistill', 'BEVDepthDistill', 'BEVDepth4DDistill',
    'LidarFormer', 'MVPFormer', 'BEVFormer', 'BEVFormerDistill'
]
