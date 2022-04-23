# 姿态估计汇总，根据MMPose修改

## 文件结构
```
 |── config               配置文件
 |    └── _base_          数据集基本配置
 |    └── body            人体身体姿态数据集配置
 |
 |── data                 数据配置
 |    └── coco            coco数据集框
 |
 |── docker               docker容器配置
 |    └── Dockfile      
 |
 |── docs                 文件教程
 |    └── zh_cn           中文教程
 |
 |── mypose               项目核心文件
 |    └── apis            API
 |    └── core            一些核心组件
 |    └── datasets        数据集处理
 |    └── models          模型配置
 |    └── utils           工具组件
 |    └── version.py      项目版本
 |
 |── tools                一些常用工具
 |    └── analysis        一些分析工具
 |    └── train.py        训练脚本   
 |
 |── README.md            readme
 |── requirments.txt      环境配置
```