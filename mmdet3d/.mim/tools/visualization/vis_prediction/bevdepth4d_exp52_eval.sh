./tools/dist_test.sh ../BEVDet/configs/lidar2camera_bev_distillation/centerpoint_pillar_to_bevdepth4d_r50/centerpoint_02pillar_second_secfpn_circlenms_8x4_cyclic_20e_nus_to_bevdepth4d_r50_virtual.py \
../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/epoch_24.pth 1 \
--cfg-options model.img_bev_encoder_neck.extra_norm_act=True \
data.val.prev_only=True data.test.prev_only=True \
model.distill_params.adaptation_type=['upsample_2layer','upsample_2layer','1x1conv'] \
model.distill_params.student_adaptation_params.kernel_size=1 \
model.distill_params.student_adaptation_params.stride=1 \
model.distill_params.student_adaptation_params.upsample_factor=4 \
model.distill_params.student_channels=[256,512,256] model.distill_params.teacher_channels=[128,256,384] \
--out ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/results_eval_valset.pkl \
--eval-options 'jsonfile_prefix=../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/results_eval_valset' \
--eval mAP
