export PYTHONPATH=$PWD:$PYTHONPATH
# bevdepth4d baseline
# python /mnt/vepfs/ML/Users/zeyu/Code/BEVDet_Distill-zeyu_lidar_distill/tools/analysis_tools/vis.py \
# ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/results_eval_valset/pts_bbox/results_nusc.json \
# --bev-draw-gt \
# --format image \
# --scale-factor 2 \
# --vis-thred 0.535 \
# --start-vis-frames 4300 \
# --end-vis-frames 4320 \
# --save_path ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/results_eval_valset/vis/ \
# --format image

python /mnt/vepfs/ML/ml-users/dingwen/code/BEVDistill_org/tools/analysis_tools/vis.py \
/mnt/vepfs/ML/ml-users/dingwen/outputs/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/results_eval_valset/pts_bbox/results_nusc.json \
--bev-draw-gt \
--format image \
--scale-factor 2 \
--vis-thred 0.535 \
--start-vis-frames 4300 \
--end-vis-frames 4320 \
--save_path /mnt/vepfs/ML/ml-users/dingwen/outputs/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/results_eval_valset/vis/ \
--format image

# bevdepth4d distilled
#[0,20]
#[100,120]
#[200,220]
#[300, 320] # may contain useful unlabeled but detected objects
#[400, 420]
#[500, 520]
#[600, 620] # might contain useful fp
#[700,720]
#[800,820]
#[1000,1020]
#[1100,1120]
#[1200,1220]
#[1300,1320]
#[1400,1420]
#[1500,1520]
#[1700,1720]
#[1800,1820]
#[1900,1920]
#[2600,2620]
#[3300,3310]
#[3400,3410]
#[3700,3710]
#[4100, 4120]
#[4200, 4220]
#[4300, 4320] # may contain useful hard example
#[4360, 4380] # may contain useful hard example
#[4400,4420]
#[4500,4520] # too many cars in a parking lot
#[4600,4620]
#[4700,4720]
#[4800,4820]
#[4900,4920] # may contain useful hard examples
#[5000, 5020]
#[5100,5120]
#[5200,5220]
#[5300,5320]
#[5400,5420]
#[5500,5520] # night, possibly all night data afterwards
#python /mnt/vepfs/ML/Users/zeyu/Code/BEVDet_Distill-zeyu_lidar_distill/tools/analysis_tools/vis.py \
#../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/results_eval_valset/pts_bbox/results_nusc.json \
#--save_path ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/results_eval_valset/vis \
#--bev-draw-gt \
#--format image \
#--scale-factor 2 \
#--vis-thred 0.535 \
#--start-vis-frames 4300 \
#--end-vis-frames 4320

# python /mnt/vepfs/ML/ml-users/dingwen/code/BEVDistill_org/tools/analysis_tools/vis.py \
# /mnt/vepfs/ML/ml-users/dingwen/outputs/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/results_eval_valset/pts_bbox/results_nusc.json \
# --save_path /mnt/vepfs/ML/ml-users/dingwen/outputs/exp52_mvp_hh1x1conv_b1b1b2b2_up4_2layer_fgd_distill_teacher_student_gt_not_combinegt_a6e-3b4e-2_teacher_fp_head_0.1_6e-2_curr_mn5nt2_inherithead_2nodes/results_eval_valset/vis \
# --bev-draw-gt \
# --format image \
# --scale-factor 2 \
# --vis-thred 0.535 \
# --start-vis-frames 4300 \
# --end-vis-frames 4320