# Training and Evaluation
After preparing data, run the following scripts to train the model. By default, all models are trained on 8 GPUs.
## BEVDepth
Train a distilled BEVDepth with CenterPoint as teacher
```
./scripts/teacher_to_bevdepth4d/centerpoint2bevdepth.sh
```
Train a distilled BEVDepth with MVP as teacher
```
./scripts/teacher_to_bevdepth4d/mvp2bevdepth.sh
```
## BEVFormer
Train a distilled BEVFormer with CenterPoint as teacher
```
./scripts/teacher_to_bevformer/exp_lidar_r50.sh
```
Train a distilled BEVFormer with MVP as teacher
```
./scripts/teacher_to_bevformer/exp_mvp_r50.sh
```
## Evaluation
For single-GPU evaluation, you can run following command
```
python tools/test.py configs/lidar2camera_bev_distillation/teacher_to_bevformer/lidarformer_to_bevformer_nus_1x1conv_r50.py outputs/lidarformer_to_bevformer_r50/epoch_24.pth --eval mAP
```
