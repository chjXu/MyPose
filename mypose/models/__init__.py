# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # 骨干网络
from .builder import (BACKBONES, HEADS, LOSSES, MESH_MODELS, NECKS, POSENETS,
                      build_backbone, build_head, build_loss, build_mesh_model,
                      build_neck, build_posenet)
from .detectors import *  # 检测器
from .heads import *  # 网络输出
from .losses import *  # 损失函数
from .necks import *  # 全局平均池化函数
# from .utils import *  # noqa

__all__ = [
    'BACKBONES', 'HEADS', 'NECKS', 'LOSSES', 'POSENETS', 'MESH_MODELS',
    'build_backbone', 'build_head', 'build_loss', 'build_posenet',
    'build_neck', 'build_mesh_model'
]