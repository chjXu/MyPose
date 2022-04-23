# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn import build_model_from_cfg
from mmcv.utils import Registry

MODELS = Registry(
    'models', build_func=build_model_from_cfg, parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
POSENETS = MODELS
MESH_MODELS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


# 构建姿态网络
# 输入cfg.model，如下所示：
'''
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    keypoint_head=dict(
        type='DeepposeRegressionHead',
        in_channels=2048,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='SmoothL1Loss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(flip_test=True))
'''
def build_posenet(cfg):
    """Build posenet."""
    return POSENETS.build(cfg)


def build_mesh_model(cfg):
    """Build mesh model."""
    return MESH_MODELS.build(cfg)