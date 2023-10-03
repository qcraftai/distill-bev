_base_ = [
    '../centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py',
]

# pay attention to this when base point cloud range is changed!
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model = dict(
    type='DynamicCenterPoint',
    pts_voxel_layer=dict(
        max_num_points=-1, max_voxels=(-1, -1),
    ),
    pts_voxel_encoder=dict(
        _delete_=True,
        type='DynamicPillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
)

