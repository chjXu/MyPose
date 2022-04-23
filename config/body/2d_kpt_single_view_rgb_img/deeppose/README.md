# DeepPose: Human pose estimation via deep neural networks

## Introduction

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

DeepPose首次提出使用深度神经网络（DNNs）解决人体姿态估计问题。它采取自定向下的策略，首先检测的bounding boxes，然后直接回归人体关节坐标（姿态）。