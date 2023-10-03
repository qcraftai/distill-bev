# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

from mmcv.utils import TORCH_VERSION, digit_version
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook

kd_loss_names = ['kd_feat_loss', 'kd_fg_feat_loss', 'kd_bg_feat_loss', 'kd_fp_bg_feat_loss'
                 'kd_channel_loss', 'kd_spatial_loss', 'kd_nonlocal_loss', 'kd_affinity_loss',
                 'kd_heatmap_loss', 's2m2_ssd_heatmap_kd_loss', 's2m2_ssd_feature_kd_loss',
                's2m2_ssd_feature_kd_tp_loss', 's2m2_ssd_feature_kd_fp_loss', 's2m2_ssd_feature_kd_fn_loss'
                 'task0_kd_heatmap_loss', 'task1_kd_heatmap_loss', 'task2_kd_heatmap_loss',
                 'task3_kd_heatmap_loss', 'task4_kd_heatmap_loss', 'task5_kd_heatmap_loss']
student_suffixes = ['head', 'lss', 'backbone0', 'backbone1', 'backbone2']
teacher_suffixes = ['head', 'lss', 'backbone0', 'backbone1', 'backbone2']
kd_loss_names_with_suffixes = [kd_loss_name + '_' + student_suffix + '_' + teacher_suffix
                               for kd_loss_name in kd_loss_names
                               for student_suffix in student_suffixes
                               for teacher_suffix in teacher_suffixes]

@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

        #############
        try:
            total_loss = tags['train/loss']
            flag = False
            for name in (kd_loss_names + kd_loss_names_with_suffixes):
                if ('train/' + name) in tags.keys():
                    flag = True
                    total_loss = total_loss - tags['train/' + name]
            if flag:
                self.writer.add_scalar('train/loss_without_kd', total_loss, self.get_iter(runner))
        except:
            print(f'tags: {tags}')
        ############

    @master_only
    def after_run(self, runner) -> None:
        self.writer.close()
