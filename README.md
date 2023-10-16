## DistillBEV: Boosting Multi-Camera 3D Object Detection with Cross-Modal Knowledge Distillation
<p align='left'>
  <img src='docs/teaser.png' width='690'/>
</p>

[Zeyu Wang](https://zw615.github.io/), [Dingwen Li](https://sites.google.com/site/dingwenli93), [Chenxu Luo](https://chenxuluo.github.io/), [Cihang Xie](https://cihangxie.github.io/), [Xiaodong Yang](https://xiaodongyang.org/) <br>
[DistillBEV: Boosting Multi-Camera 3D Object Detection with Cross-Modal Knowledge Distillation](https://arxiv.org/pdf/2309.15109.pdf), ICCV 2023 <br>

## Get Started

### Installtion
Please refer to [INSTALL](docs/INSTALL.md) to set up the environment and install the dependencies (see more details in [Dockerfile](docker/Dockerfile)).

### Data Preparation
Please follow the instructions in [DATA](docs/DATA.md).

### Training and Evaluation
Please follow the instructions in [RUN](docs/RUN.md).

## Main Results

BEVDepth and BEVFormer are respectively used in this repo as CNNs and Transformers based students to exemplify the improvement by DistillBEV. 

* Backbone: ResNet-50 pre-trained on ImageNet-1K 

| Teacher | Student | mAP | NDS |
| ------- | ------- | --- | --- |
| - | BEVDepth | 36.4	| 48.4 |
| CenterPoint | BEVDepth | 38.9	| 49.8 |
| MVP | BEVDepth | 40.3	| 51.0 |

| Teacher | Student | mAP | NDS |
| ------- | ------- | --- | --- |
| - | BEVFormer | 32.3 | 43.4 |
| CenterPoint | BEVFormer | 35.5 | 46.8 |
| MVP | BEVFormer | 36.7 | 47.6 |

## Citation
Please cite the following paper if this repo helps your research: <br>
```
@InProceedings{Wang_2023_ICCV,
    author    = {Wang, Zeyu and Li, Dingwen and Luo, Chenxu and Xie, Cihang and Yang, Xiaodong},
    title     = {DistillBEV: Boosting Multi-Camera 3D Object Detection with Cross-Modal Knowledge Distillation},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
```

## Acknowledgement
We thank the authors for the multiple great open-sourced repos, including [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) and [BEVFormer](https://github.com/fundamentalvision/BEVFormer).

## License
Copyright (C) 2023 QCraft. All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (Attribution-NonCommercial-ShareAlike 4.0 International). The code is released for academic research use only. For commercial use, please contact business@qcraft.ai.
