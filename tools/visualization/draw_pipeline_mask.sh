# based on bevdepth4d exp52.sh
# the selected is 1000
./tools/dist_train.sh ../BEVDet/configs/lidar2camera_bev_distillation/centerpoint_pillar_to_bevdepth4d_r50/centerpoint_02pillar_second_secfpn_circlenms_8x4_cyclic_20e_nus_to_bevdepth4d_r50_virtual.py 1 \
--cfg-options model.inherit_head=True \
model.img_bev_encoder_neck.extra_norm_act=True \
data.val.prev_only=True data.test.prev_only=True \
model.teacher_config='../BEVDet/configs/mvp/mvp_dynamic_centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py' \
model.teacher_ckpt='../BEVDet/output/mvp/exp2_mvp_dynamic_centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/epoch_20.pth' \
model.distill_params.spatial_attentions=['teacher_student',] \
model.distill_params.foreground_mask='gt' model.distill_params.background_mask='logical_not' \
model.distill_params.scale_mask='combine_gt' \
model.distill_params.adaptation_type=['upsample_2layer','upsample_2layer','1x1conv'] \
model.distill_params.student_adaptation_params.kernel_size=1 \
model.distill_params.student_adaptation_params.stride=1 \
model.distill_params.student_adaptation_params.upsample_factor=4 \
model.distill_params.student_channels=[256,512,256] model.distill_params.teacher_channels=[128,256,384] \
model.distill_params.student_feat_pos=['backbone1','backbone2','head'] model.distill_params.teacher_feat_pos=['backbone1','backbone2','head'] \
model.distill_params.fp_as_foreground=['teacher','teacher','teacher'] model.distill_params.output_threshold=0.1 \
model.distill_params.fp_weight=6e-2 model.distill_params.fp_scale_mode='average' \
model.distill_params.fg_feat_loss_weights=[6e-3,] \
model.distill_params.bg_feat_loss_weights=[4e-2,] \
model.distill_params.spatial_t=0.5 \
optimizer_config._delete_=True optimizer_config.grad_clip.max_norm=5 optimizer_config.grad_clip.norm_type=2 \
optimizer.lr=4e-4 \
data.samples_per_gpu=8 \
data.train.type='IdentityDataset' \
data.train.dataset.select_index=1000 model.distill_params.save_attention=True \
runner.max_epochs=1 \
--checkpoint ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/epoch_24.pth
#--checkpoint ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/epoch_24.pth
#--resume-from ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/epoch_24.pth


