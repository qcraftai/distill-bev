# ok:
# 【450, 550] 0.5, might need to change
# [673,711] 0.5
# [1110, 1188] 0.5
# [1398, 1426] 0.25. Needs selection
# [1550, 1584] 0.5
# [1825， 2183】 0.3. Needs selection
# [3150, 3391] 0.2.  Needs selection

# not ok:
# [850, 950]
# [1050,1109]
# [1189,1250]
# [1350, 1450] a fn even with thres 0.25. some are ok. Decide to cherry pick...
# [1584, 1650]
export PYTHONPATH=$PWD:$PYTHONPATH
# bevdepth4d distilled test set best version
python /mnt/vepfs/ML/Users/zeyu/Code/BEVDet_Distill-zeyu_lidar_distill/tools/analysis_tools/vis.py \
./output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d_testset/exp14_bevdepth4d_swin_base_testlb_2_trainval_mvp_trainval_voxel1e-1_10sweeps_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_withc_a1.5e-3b2e-2sw6e-4_ow1.2vw1.2_teacher_fp_head_0.1_6e-2_ih_curr_mn5nt2_bsz2_lr4e-4_8nodes/results_eval_epoch23/pts_bbox/results_nusc.json \
--save_path ./output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d_testset/exp14_bevdepth4d_swin_base_testlb_2_trainval_mvp_trainval_voxel1e-1_10sweeps_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_withc_a1.5e-3b2e-2sw6e-4_ow1.2vw1.2_teacher_fp_head_0.1_6e-2_ih_curr_mn5nt2_bsz2_lr4e-4_8nodes/results_eval_valset/vis \
--format image \
--canva-size 450 \
--scale-factor 4 \
--vis-thred 0.2 \
--start-vis-frames 5450 \
--end-vis-frames 5550 \
--version test