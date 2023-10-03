# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import numpy as np
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--save_attention', action='store_true', default=False)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    ######################
    if args.save_attention:
    # if gt_bboxes is required but wrapped in 'MultiScaleFlipAug3D'
    # the 'MultiScaleFlipAug3D' actually means to wrap data with a list. i.e. data[key] = list(data[key])
    # so it fits into self.simple_test(points[0], img_metas[0], img[0], **kwargs) in forward_test in Base3DDetector
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = False
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = False
        assert 'BEVDet' in cfg.model.type or 'BEVDepth' in cfg.model.type
        assert 'centerpoint' in cfg.model.teacher_config

        cfg.data.test.pipeline = [
            dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=cfg.data_config),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=cfg.file_client_args),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=9,
                use_dim=[0, 1, 2, 3, 4],
                file_client_args=cfg.file_client_args,
                pad_empty_sweeps=True,
                remove_close=True),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            # since we are using training mode to read nuscenes data, we don't need 'MultiScaleFlipAug3D' to wrap data with a list
            # otherwise example['gt_labels_3d']._data in prepare_train_data will throw an error
            dict(
                type='DefaultFormatBundle3D',
                class_names=cfg.class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ]
        if '4D' in cfg.model.type:
            cfg.data.test.pipeline[0] = dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=cfg.data_config,
                     sequential=True, aligned=True, trans_only=False,
                     root_path=cfg.data_root)
        if 'BEVDepth' in cfg.model.type:
            cfg.data.test.pipeline.insert(2, dict(type='PointToMultiViewDepth', grid_config=cfg.grid_config))
    ####################
    dataset = build_dataset(cfg.data.test)
    # if args.save_attention:
    #     dataset.flag = np.zeros(len(dataset))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False if not args.save_attention else True)
    # from mmcv.parallel import collate
    # from functools import partial
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=samples_per_gpu,
    #     num_workers=cfg.data.workers_per_gpu,
    #     pin_memory=True,
    #     shuffle=False,
    #     collate_fn=partial(collate, samples_per_gpu=samples_per_gpu))
    # data = iter(data_loader).next()

    # build the model and load checkpoint
    if not args.save_attention:
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    else:
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # FIXME with syncbn!
    try:
        from mmcv.cnn.utils import revert_sync_batchnorm
        model = revert_sync_batchnorm(model)
    except:
        pass
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    #############
    # FIXME debug only
    # from mmcv.parallel import MMDataParallel
    # from copy import deepcopy
    # from mmdet3d.apis import single_gpu_test
    # rank, _ = get_dist_info()
    #
    # class_names = [
    #     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    #     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    # ]
    # file_client_args = dict(backend='disk')
    # cfg.data.test.pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points'])
    #     ])
    # ]
    #
    # if rank == 1:
    #     dataset = build_dataset(cfg.data.test)
    #     data_loader = build_dataloader(
    #         dataset,
    #         samples_per_gpu=1,
    #         workers_per_gpu=8,
    #         dist=False,
    #         shuffle=False)
    #     teacher_model = deepcopy(model.teacher_model)
    #     teacher_model = MMDataParallel(teacher_model, device_ids=[0])
    #     outputs = single_gpu_test(teacher_model, data_loader)
    #
    #
    #     eval_kwargs = cfg.get('evaluation', {}).copy()
    #     # hard-code way to remove EvalHook args
    #     for key in [
    #         'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
    #         'rule'
    #     ]:
    #         eval_kwargs.pop(key, None)
    #     eval_kwargs.update(dict(metric='mAP'))
    #     print(dataset.evaluate(outputs, **eval_kwargs))
    #     import pdb
    #     pdb.set_trace()
    #
    # import torch.distributed as dist
    # dist.barrier()
    #
    # import sys
    # sys.exit()
    ############

    rank, world_size = get_dist_info()
    if not distributed or world_size==1:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  save_attention=args.save_attention, warp_list=args.save_attention)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
