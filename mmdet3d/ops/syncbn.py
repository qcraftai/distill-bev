from mmcv.ops import SyncBatchNorm
from mmcv.cnn import NORM_LAYERS
from torch.nn import init

@NORM_LAYERS.register_module(name='MMSyncBNOneInit')
class SyncBatchNormOneInit(SyncBatchNorm):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            # self.weight.data.ones_()  # pytorch use ones_()
            # self.bias.data.zero_()
            init.ones_(self.weight)
            init.zeros_(self.bias)