./tools/dist_test.sh ../BEVDet/configs/bevdepth/bevdepth4d-r50.py \
../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/epoch_24.pth 1 \
--cfg-options model.img_bev_encoder_neck.extra_norm_act=True \
data.val.prev_only=True data.test.prev_only=True \
--out ../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/results_eval_valset.pkl \
--eval-options 'jsonfile_prefix=../BEVDet/output/lidar2camera_bev_distillation/centerpoint_to_bevdepth4d/exp6_bevdepth4d_r50_curr_for_no_prior_extrana/results_eval_valset' \
--eval mAP
