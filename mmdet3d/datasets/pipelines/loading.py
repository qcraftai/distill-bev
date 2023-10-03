# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import torchvision
from PIL import Image
import mmcv
import numpy as np
from pyquaternion import Quaternion
from functools import partial

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=16):
        self.downsample = downsample
        self.grid_config=grid_config

    def points2depthmap(self, points, height, width, canvas=None):
        height, width = height//self.downsample, width//self.downsample
        depth_map = torch.zeros((height,width), dtype=torch.float32)
        coor = torch.round(points[:,:2]/self.downsample)
        depth = points[:,2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
               & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth < self.grid_config['dbound'][1]) \
                & (depth >= self.grid_config['dbound'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks+depth/100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:,1],coor[:,0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        # imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs']
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
        depth_map_list = []
        for cid in range(rots.shape[0]):
            combine = rots[cid].matmul(torch.inverse(intrins[cid]))
            combine_inv = torch.inverse(combine)
            points_img = (points_lidar.tensor[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
            points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                   points_img[:,2:3]], 1)
            points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        # results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        results['img_inputs'] = results['img_inputs'] + (depth_map,)
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

def bevdepth_normalize(img, img_mean, img_std, to_rgb):
    img = mmcv.imnormalize(np.array(img), img_mean, img_std, to_rgb)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_BEVDet(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True,
                 root_path='./data/nuscenes',
                 bevdepth_norm=False):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]))) if not bevdepth_norm else \
            partial(bevdepth_normalize,
                img_mean=np.array([[123.675, 116.28, 103.53]]),
                img_std=np.array([58.395, 57.12, 57.375]), to_rgb=True)

        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only

        self._root_path = root_path

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize # A1 * x
        post_tran -= torch.Tensor(crop[:2]) # A1 * x + b1
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b # A2 * (A1 * x + b1) + b2
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        # move centerpoint to origin, rotate around origin, and then move back centerpoint
        # equivalent to rotate around the center point (crop_w, crop_h, crop_w + fW/2, crop_h + fH/2) of the image
        post_tran = A.matmul(post_tran) + b # A3 * ((A2 * (A1 * x + b1) + b2) - b3) + b3

        # post_rot and post tran are used for restore right image/BEV frustum feature in ViewTransformerLiftSplatShoot
        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            # FIXME change back!
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            # crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # FIXME change back!
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            # flip = self.data_config['flip']
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self,results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        # cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            ####################
            filename = os.path.join(os.path.abspath(self._root_path),
                                    cam_data['data_path'][cam_data['data_path'].find('samples'):])
            ####################
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    if self.return_sensor2ego:
                        raise NotImplementedError
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego

                        ###############
                        # no prior frame. deuplicate current frame
                        if results['adjacent_type'] == 'curr':
                            # assert np.allclose(lidaradj2lidarcurr, np.eye(4))
                            lidaradj2lidarcurr = np.eye(4, dtype=np.float32)
                        ###############
                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_BEVDepth(LoadMultiViewImageFromFiles_BEVDet):
    def __init__(self, bevdepth_norm=False, return_sensor2ego=False, **kwargs):
        super(LoadMultiViewImageFromFiles_BEVDepth, self).__init__(**kwargs)
        self.bevdepth_norm = bevdepth_norm
        self.return_sensor2ego = return_sensor2ego
        if self.bevdepth_norm:
            self.normalize_img = partial(bevdepth_normalize,
                                         img_mean=np.array([[123.675, 116.28, 103.53]]),
                                         img_std=np.array([58.395, 57.12, 57.375]), to_rgb=True)

    # return sensor to ego
    # more accurate lidaradj2lidarcurr for sweep images
    def get_inputs(self,results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        sensor2ego_rots = []
        sensor2ego_trans = []
        cams = self.choose_cams()
        # cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            ####################
            filename = os.path.join(os.path.abspath(self._root_path),
                                    cam_data['data_path'][cam_data['data_path'].find('samples'):])
            ####################
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            ##################
            w, x, y, z = cam_data['sensor2ego_rotation']
            # sweep sensor to sweep ego
            sensor2ego_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sensor2ego_tran = torch.Tensor(
                cam_data['sensor2ego_translation'])
            sensor2ego = sensor2ego_rot.new_zeros(
                (4, 4))
            sensor2ego[3, 3] = 1
            sensor2ego[:3, :3] = sensor2ego_rot
            sensor2ego[:3, -1] = sensor2ego_tran

            assert torch.all(sensor2ego[-1] == torch.tensor([0., 0., 0., 1.]))
            # sensor2egos.append(sensor2ego)
            sensor2ego_rots.append(sensor2ego[:3, :3])
            sensor2ego_trans.append(sensor2ego[:3, 3])
            ####################

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                raise NotImplementedError
                # if not type(results['adjacent']) is list:
                #     rots.extend(rots)
                #     post_trans.extend(post_trans)
                #     post_rots.extend(post_rots)
                #     intrins.extend(intrins)
                #     if self.aligned:
                #         posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                #         posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                #         shift_global = posi_adj - posi_curr
                #
                #         l2e_r = results['curr']['lidar2ego_rotation']
                #         e2g_r = results['curr']['ego2global_rotation']
                #         l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                #         e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                #
                #         # shift_global = np.array([*shift_global[:2], 0.0])
                #         shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                #             l2e_r_mat).T
                #         trans.extend([tran + shift_lidar for tran in trans])
                #     else:
                #         trans.extend(trans)
                # else:
                #     assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        trans_new = []
                        rots_new =[]
                        for cam, tran,rot in zip(cams, trans, rots):
                            ###############
                            # no prior frame. deuplicate current frame
                            if results['adjacent_type'] == 'curr':
                                # assert np.allclose(lidaradj2lidarcurr, np.eye(4))
                                lidaradj2lidarcurr = np.eye(4, dtype=np.float32)
                            else:
                                egoadj2global = np.eye(4, dtype=np.float32)
                                egoadj2global[:3, :3] = Quaternion(
                                    results['adjacent']['cams'][cam]['ego2global_rotation']).rotation_matrix
                                egoadj2global[:3, 3] = results['adjacent']['cams'][cam]['ego2global_translation']

                                lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                                     @ egoadj2global @ lidar2ego

                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)

                        ##################
                        for cam in cams:
                            w, x, y, z = results['adjacent']['cams'][cam]['sensor2ego_rotation']
                            # sweep sensor to sweep ego
                            sweepsensor2sweepego_rot = torch.Tensor(
                                Quaternion(w, x, y, z).rotation_matrix)
                            sweepsensor2sweepego_tran = torch.Tensor(
                                results['adjacent']['cams'][cam]['sensor2ego_translation'])
                            sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                                (4, 4))
                            sweepsensor2sweepego[3, 3] = 1
                            sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                            sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran

                            if results['adjacent_type'] == 'curr':
                                sweepsensor2keyego = sweepsensor2sweepego
                            else:
                                # global 2 key ego
                                cam_data = results['img_info'][cam]
                                # global sensor to cur ego
                                w, x, y, z = cam_data['ego2global_rotation']
                                keyego2global_rot = torch.Tensor(
                                    Quaternion(w, x, y, z).rotation_matrix)
                                keyego2global_tran = torch.Tensor(
                                    cam_data['ego2global_translation'])
                                keyego2global = keyego2global_rot.new_zeros((4, 4))
                                keyego2global[3, 3] = 1
                                keyego2global[:3, :3] = keyego2global_rot
                                keyego2global[:3, -1] = keyego2global_tran
                                global2keyego = keyego2global.inverse()

                                # sweep ego to global
                                w, x, y, z = results['adjacent']['cams'][cam]['ego2global_rotation']
                                sweepego2global_rot = torch.Tensor(
                                    Quaternion(w, x, y, z).rotation_matrix)
                                sweepego2global_tran = torch.Tensor(
                                    results['adjacent']['cams'][cam]['ego2global_translation'])
                                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                                sweepego2global[3, 3] = 1
                                sweepego2global[:3, :3] = sweepego2global_rot
                                sweepego2global[:3, -1] = sweepego2global_tran

                                sweepsensor2keyego = global2keyego @ sweepego2global @ \
                                                     sweepsensor2sweepego
                            # sensor2egos.append(sweepsensor2keyego)
                            assert torch.all(sweepsensor2keyego[-1] == torch.tensor([0., 0., 0., 1.]))
                            sensor2ego_rots.append(sweepsensor2keyego[:3, :3])
                            sensor2ego_trans.append(sweepsensor2keyego[:3, 3])

                    else:
                        # rots.extend(rots)
                        # trans.extend(trans)
                        raise NotImplementedError
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans, sensor2ego_rots, sensor2ego_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans),
                                                            torch.stack(sensor2ego_rots), torch.stack(sensor2ego_trans))
        if self.return_sensor2ego:
            return imgs, rots, trans, intrins, post_rots, post_trans, sensor2ego_rots, sensor2ego_trans
        else:
            return imgs, rots, trans, intrins, post_rots, post_trans

@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
        interval (int): the interval to sample sweeps from (0, sweeps_num)
        multi_sweeps (list of int): sample another sweeps_num sweeps from (multi_sweeps[i], len(results['sweeps']))
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 interval=None,
                 multi_sweeps=None,
                 drop_rate=0.,
                 virtual=False,
                 pseudo_virtual=False,):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.interval = interval
        self.multi_sweeps = multi_sweeps
        self.drop_rate = drop_rate
        self.virtual = virtual
        self.pseudo_virtual = pseudo_virtual
        assert not (self.virtual and self.pseudo_virtual)
        print(f'using drop_rate {self.drop_rate} in loading multi sweeps')
        if self.virtual:
            print(f'using virtual points')
        if self.pseudo_virtual:
            print(f'using pseudo virtual points')



    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            if '/mnt/vepfs/ML/ml-public/nuscenes_virtual/' in pts_filename:
                pts_filename = pts_filename.replace('/mnt/vepfs/ML/ml-public/nuscenes_virtual/', '/mnt/vepfs/ML/ml-public/Stuff/nuscenes_virtual/')
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        # points.tensor[:, 4] = 0
        points.tensor[:, -1] = 0
        sweep_points_list = [points]
        ts = results['timestamp'] # timestamp already / 1e6
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            sweep_num = self.sweeps_num * (1 + len(self.multi_sweeps)) if self.multi_sweeps is not None else self.sweeps_num
            for i in range(sweep_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if self.interval is not None:
                assert isinstance(self.interval, int)
                if len(results['sweeps']) <= self.sweeps_num // self.interval:
                    choices = np.arange(len(results['sweeps']))
                # elif len(results['sweeps']) <= self.sweeps_num and not self.test_mode:
                #     choices = np.random.choice(
                #         len(results['sweeps']), self.sweeps_num // self.interval, replace=False)
                elif len(results['sweeps']) <= self.sweeps_num:
                    choices = np.arange(self.sweeps_num // self.interval)
                else:
                    choices = np.arange(0, self.sweeps_num, self.interval)
            else:
                # a path to keep old behaviour
                if len(results['sweeps']) <= self.sweeps_num:
                    choices = np.arange(len(results['sweeps']))
                elif self.test_mode:
                    choices = np.arange(self.sweeps_num)
                else:
                    choices = np.random.choice(
                        len(results['sweeps']), self.sweeps_num, replace=False)

            if self.multi_sweeps is not None:
                # TODO: use sweep num +1
                assert isinstance(self.interval, int)
                assert isinstance(self.multi_sweeps, list) and len(self.multi_sweeps) > 0
                for val in self.multi_sweeps:
                    assert isinstance(val, int)

                multi_sweeps_choices_list = []
                for start_sweep in self.multi_sweeps:
                    assert start_sweep >= self.sweeps_num
                    if 'adjacent' in results:
                        assert not isinstance(results, list)
                        assert results['adjacent_type'] != 'next'
                        # FIXME for now, this implementation only suits nuscenes two-frame fusion
                        timestamp_front = results['adjacent']['timestamp']
                        timestamp_list = np.array([sweep['timestamp'] for sweep in results['sweeps']], dtype=np.long)
                        diff = np.abs(timestamp_list - timestamp_front) / 1e6
                        start_sweep = np.argsort(diff)[0]

                    if len(results['sweeps']) <= start_sweep:
                        multi_sweeps_choices = np.array([], dtype=int)
                    elif len(results['sweeps']) <= start_sweep + self.sweeps_num // self.interval:
                        multi_sweeps_choices = np.arange(start_sweep, len(results['sweeps']), 1)
                    elif len(results['sweeps']) <= start_sweep + self.sweeps_num:
                        multi_sweeps_choices = np.arange(start_sweep, start_sweep + self.sweeps_num // self.interval, 1)
                    else:
                        # TODO: test if test_mode is ever turned on
                        if 'adjacent' in results:
                            pass
                        elif self.test_mode:
                            start_sweep = min(int((start_sweep + len(results['sweeps'])) / 2),
                                              len(results['sweeps']) - self.sweeps_num)
                        else:
                            # double exclusive, compensate with +1
                            start_sweep = np.random.randint(start_sweep, len(results['sweeps']) - self.sweeps_num + 1)
                        multi_sweeps_choices = np.arange(start_sweep, start_sweep + self.sweeps_num, self.interval)

                    # for subsequent transforms, must be concated with current points
                    multi_sweeps_choices_list.append(multi_sweeps_choices)

                # repeat sweep read code
                # if self.separate_multi_sweeps:
                #     multi_points_list = []
                #     for multi_sweeps_choices in multi_sweeps_choices_list:
                #         multi_sweep_points_list = []
                #         for idx in multi_sweeps_choices:
                #             sweep = results['sweeps'][idx]
                #             points_sweep = self._load_points(sweep['data_path'])
                #             points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                #             if self.remove_close:
                #                 points_sweep = self._remove_close(points_sweep)
                #             sweep_ts = sweep['timestamp'] / 1e6
                #             points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                #                 'sensor2lidar_rotation'].T
                #             points_sweep[:, :3] += sweep['sensor2lidar_translation']
                #             points_sweep[:, 4] = ts - sweep_ts
                #             points_sweep = points.new_point(points_sweep)
                #             multi_sweep_points_list.append(points_sweep)
                #         multi_points = points.cat(multi_sweep_points_list)
                #         multi_points = multi_points[:, self.use_dim]
                #         multi_points_list.append(multi_points)
                # else:
                for multi_sweeps_choices in multi_sweeps_choices_list:
                    choices = np.concatenate((choices, multi_sweeps_choices))

            keeps = np.random.uniform(low=0., high=1., size=len(choices)) >= self.drop_rate
            choices = choices[keeps]

            # no need to shuffle here. PointShuffle will do it
            # count = 0
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                #########################
                if self.virtual:
                    # direct copy from centerpoint
                    # WARNING: hard coded for nuScenes
                    # remove dim for beam number (dim 4 in nuscenes)
                    assert self.use_dim == list(range(17))
                    points_sweep = np.concatenate([points_sweep,
                                                   np.ones([points_sweep.shape[0], 15 - points_sweep.shape[1]]),
                                                   # cls label
                                                   np.ones([points_sweep.shape[0], 1]),  # virtual label dim
                                                   np.zeros([points_sweep.shape[0], 1])], axis=1)  # timestamp dim

                    tokens = sweep['data_path'].split('/')
                    # seg_path = os.path.join('/', *tokens[:-2], tokens[-2] + "_VIRTUAL", tokens[-1] + '.pkl.npy')
                    seg_path = os.path.join(*tokens[:-2], tokens[-2] + "_VIRTUAL", tokens[-1] + '.pkl.npy')
                    # for missing sweeps
                    # 295489 sweeps in MVP vs 297737 sweeps in Nuscenes Train val Set
                    try:
                        data_dict = np.load(seg_path, allow_pickle=True).item()
                    except:
                        # count += 1
                        # if count > 5:
                        #     import pdb
                        #     pdb.set_trace()
                        continue

                    # virtual_points1.shape[0] != points.shape[0]
                    virtual_points1 = data_dict['real_points']  # [:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
                    virtual_points2 = data_dict['virtual_points']
                    # insert -1 for reflectance as virtual points2 don't have this value
                    virtual_points2 = np.concatenate(
                        [virtual_points2[:, [0, 1, 2]], -1 * np.ones([virtual_points2.shape[0], 1]),
                         virtual_points2[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], axis=1)

                    virtual_points1 = np.concatenate([virtual_points1,
                                                      np.zeros([virtual_points1.shape[0], 1]),  # virtual label dim
                                                      np.zeros([virtual_points1.shape[0], 1])], axis=1)  # timestamp dim
                    virtual_points2 = np.concatenate([virtual_points2,
                                                      -1 * np.ones([virtual_points2.shape[0], 1]),  # virtual label dim
                                                      np.zeros([virtual_points2.shape[0], 1])], axis=1)  # timestamp dim
                    points_sweep = np.concatenate([points_sweep, virtual_points1, virtual_points2], axis=0).astype(np.float32)
                    assert points_sweep.shape[1] == 17 and points.shape[1] == 17
                elif self.pseudo_virtual:
                    # only use mvp for the first sweep
                    assert self.use_dim == list(range(17))
                    points_sweep = np.concatenate([points_sweep,
                                                   np.ones([points_sweep.shape[0], 15 - points_sweep.shape[1]]),
                                                   # cls label
                                                   np.ones([points_sweep.shape[0], 1]),  # virtual label dim
                                                   np.zeros([points_sweep.shape[0], 1])], axis=1)  # timestamp dim

                    points_sweep = points_sweep.astype(np.float32)
                    assert points_sweep.shape[1] == 17 and points.shape[1] == 17
                #############################
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # points_sweep[:, 4] = ts - sweep_ts
                points_sweep[:, -1] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        # if self.multi_sweeps is not None and self.separate_multi_sweeps:
        #     [points,] + multi_points_list
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 dummy=False,
                 file_client_args=dict(backend='disk'),
                 root_path='data/nuscenes/',
                 virtual=False,
                 half_beams=False):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.dummy = dummy

        ##########
        self._root_path = root_path
        self.virtual = virtual
        self.half_beams = half_beams
        ##########

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            if '/mnt/vepfs/ML/ml-public/nuscenes_virtual/' in pts_filename:
                pts_filename = pts_filename.replace('/mnt/vepfs/ML/ml-public/nuscenes_virtual/', '/mnt/vepfs/ML/ml-public/Stuff/nuscenes_virtual/')
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if self.dummy:
            points = torch.ones([1,self.load_dim] ,dtype=torch.float32)
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None)
            results['points'] = points
            return results
        pts_filename = results['pts_filename']
        ####################
        # if pts_filename.find('samples') != -1:
        #     pts_filename = os.path.join(os.path.abspath(self._root_path),
        #                             pts_filename[pts_filename.find('samples'):])
        # elif pts_filename.find('nuscenes_gt_database') != -1:
        #     pts_filename = os.path.join(os.path.abspath(self._root_path),
        #                                 pts_filename[pts_filename.find('nuscenes_gt_database'):])
        # else:
        #     raise NotImplementedError
        ####################
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        if self.half_beams:
            points = points[::2, :]
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.virtual:
            # direct copy from centerpoint
            # WARNING: hard coded for nuScenes
            # remove dim for beam number (dim 4 in nuscenes)
            # assert self.use_dim == [0,1,2,3] or self.use_dim == list(range(17))
            if pts_filename.startswith('./data'):    
                tokens = pts_filename.split('/')
                seg_path = os.path.join(*tokens[:-2], tokens[-2] + "_VIRTUAL", tokens[-1] + '.pkl.npy')    
            else:
                tokens = pts_filename.split('/')
                seg_path = os.path.join('/', *tokens[:-2], tokens[-2] + "_VIRTUAL", tokens[-1] + '.pkl.npy')
            
            data_dict = np.load(seg_path, allow_pickle=True).item()

            # virtual_points1.shape[0] != points.shape[0]
            virtual_points1 = data_dict['real_points'] #[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
            virtual_points2 = data_dict['virtual_points']
            # insert -1 for reflectance as virtual points2 don't have this value
            virtual_points2 = np.concatenate(
                [virtual_points2[:, [0, 1, 2]], -1 * np.ones([virtual_points2.shape[0], 1]),
                 virtual_points2[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], axis=1)

            points = np.concatenate([points,
                                     np.ones([points.shape[0], 15 - points.shape[1]]), # cls label
                                     np.ones([points.shape[0], 1]),  # virtual label dim
                                     np.zeros([points.shape[0], 1])], axis=1) # timestamp dim
            virtual_points1 = np.concatenate([virtual_points1,
                                              np.zeros([virtual_points1.shape[0], 1]), # virtual label dim
                                              np.zeros([virtual_points1.shape[0], 1])], axis=1) # timestamp dim
            virtual_points2 = np.concatenate([virtual_points2,
                                              -1 * np.ones([virtual_points2.shape[0], 1]), # virtual label dim
                                              np.zeros([virtual_points2.shape[0], 1])], axis=1) # timestamp dim
            points = np.concatenate([points, virtual_points1, virtual_points2], axis=0).astype(np.float32)
            assert points.shape[1] == 17

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk'),
                 lidar_velo_to_dist=False,
                 lidar_interval=1./20):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

        self.lidar_velo_to_dist = lidar_velo_to_dist
        self.lidar_interval = lidar_interval # synchcronize with lidar_interval in nuscenes_dataset.py

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        ###############
        # designed to translate centerpoint/mvp's prediction from velocity into distance
        if self.lidar_velo_to_dist:
            sorted_timestamps = torch.unique(results['points'].tensor[:, -1], sorted=True)
            time_interval = torch.abs(sorted_timestamps[-1] - sorted_timestamps[0])
            if time_interval.item() == 0:
                time_interval = self.lidar_interval
            # absolute speed here. don't use bevdet4d anno file! it uses ego-relative speed
            bbox = results['ann_info']['gt_bboxes_3d'].tensor
            bbox[:, 7:9] = bbox[:, 7:9] * time_interval
            results['gt_time_interval'] = time_interval
            results['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
                                                                       box_dim=bbox.shape[-1],
                                                                       origin=(0.5, 0.5, 0.0))
        ###############

        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
