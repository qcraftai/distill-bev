import glob
import cv2
import numpy as np

root_path = '/mnt/e/研/UCSC/Paper/Mono3D/Code/mine/BEVDet_Distill-zeyu_lidar_distill/output/' \
            'lidar2camera_bev_distillation/centerpoint_to_bevdepth4d_testset/' \
            'exp14_bevdepth4d_swin_base_testlb_2_trainval_mvp_trainval_voxel1e-1_10sweeps_hh1x1conv_b1b1b2b2_up4_' \
            '2layer_fgd_distill_teacher_student_gt_not_combinegt_withc_a1.5e-3b2e-2sw6e-4_ow1.2vw1.2_teacher_fp_head_0.1_6e-2_ih_curr_mn5nt2_bsz2_lr4e-4_8nodes/results_eval_valset/vis/vis/'

imgs=[]
for filename in glob.glob(root_path + '*.jpg'):
# for filename in glob.glob(root_path):
    imgs.append(cv2.imread(filename))


height,width,layers=imgs[0].shape

video=cv2.VideoWriter('bevdistill_demo.avi',-1,fps=10,frameSize=(width,height))

for img in imgs:
    video.write(img)

cv2.destroyAllWindows()
video.release()


