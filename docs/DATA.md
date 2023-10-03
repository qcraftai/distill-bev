# Data Preparation
## BEVDepth
You can download nuScenes 3D detection data from the [official webset](https://www.nuscenes.org/login?prevpath=download&prevhash=) and unzip all zip files. <br>
It is recommended to symlink the dataset root to $DISTILLBEV/data.
```
distill-bev
├── configs
├── docs
├── mmdet3d
├── requirements
├── scripts
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```
First, run the following command to get the .pkl files.
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes –virtual --extra-tag nuscenes
```
```
distill-bev
├── configs
├── docs
├── mmdet3d
├── requirements
├── scripts
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
|   |   ├── nuscenes_infos_train.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_test.pkl
```
Then, we run the following command to generate the adjacent information for BEVDepth.
```
python tools/data_converter/prepare_nuscenes_for_bevdet4d.py
```
```
distill-bev
├── configs
├── docs
├── mmdet3d
├── requirements
├── scripts
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
|   |   ├── nuscenes_infos_train.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_test.pkl
|   |   ├── nuscenes_infos_train_4d_interval3_max60.pkl
|   |   ├── nuscenes_infos_val_4d_interval3_max60.pkl
|   |   ├── nuscenes_infos_test_4d_interval3_max60.pkl
```
## BEVFormer
For data preparation with BEVFormer, please refer to the official [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md) repository.
