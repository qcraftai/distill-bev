export CUDA_HOME=/usr/local/cuda # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"

filename=mvp2bevdepth.sh
logdir=/mnt/vepfs/ML/ml-users/dingwen/outputs/centerpoint_to_bevdepth4d_r50_virtual/


cd /mnt/vepfs/ML/ml-users/dingwen/code/DistillBEV
pip install -v -e .

mkdir -p ${logdir}
cat ./scripts/teacher_to_bevdepth4d/$filename > ${logdir}run.sh

############
cp ./tools/epoch_based_runner_modified.py /opt/conda/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py
cp ./tools/tensorboard_modified.py /opt/conda/lib/python3.8/site-packages/mmcv/runner/hooks/logger/tensorboard.py
############

./tools/multi_node_dist_train.sh configs/lidar2camera_bev_distillation/centerpoint_pillar_to_bevdepth4d_r50/centerpoint_02pillar_second_secfpn_circlenms_8x4_cyclic_20e_nus_to_bevdepth4d_r50_virtual.py \
--cfg-options model.inherit_head=True \
model.img_bev_encoder_neck.extra_norm_act=True \
data.val.prev_only=True data.test.prev_only=True \
model.teacher_config='configs/mvp/mvp_dynamic_centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py' \
model.teacher_ckpt='/mnt/vepfs/ML/ml-users/zeyu/BEVDet/output/mvp/exp2_mvp_dynamic_centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/epoch_20.pth' \
model.distill_params.spatial_attentions=['teacher_student',] \
model.distill_params.foreground_mask='gt' model.distill_params.background_mask='logical_not' \
model.distill_params.scale_mask='combine_gt' \
model.distill_params.adaptation_type=['upsample_3layer','upsample_3layer','1x1conv'] \
model.distill_params.student_adaptation_params.kernel_size=1 \
model.distill_params.student_adaptation_params.stride=1 \
model.distill_params.student_adaptation_params.upsample_factor=4 \
model.distill_params.student_channels=[256,512,256] model.distill_params.teacher_channels=[128,256,384] \
model.distill_params.student_feat_pos=['backbone1','backbone2','head'] model.distill_params.teacher_feat_pos=['backbone1','backbone2','head'] \
model.distill_params.fp_as_foreground=['none','none','teacher'] model.distill_params.output_threshold=0.1 \
model.distill_params.fp_weight=6e-2 model.distill_params.fp_scale_mode='average' \
model.distill_params.fg_feat_loss_weights=[6e-3,] \
model.distill_params.bg_feat_loss_weights=[4e-2,] \
model.distill_params.channel_mask=False \
model.img_backbone.pretrained='/mnt/vepfs/ML/ml-users/dingwen/pretrained/resnet50-19c8e357.pth' \
optimizer_config._delete_=True optimizer_config.grad_clip.max_norm=5 optimizer_config.grad_clip.norm_type=2 \
optimizer.lr=2e-4 \
checkpoint_config.interval=4 \
--work-dir ${logdir}

cat ./scripts/teacher_to_bevdepth4d/$filename