checkpoint_config = dict(interval=5) # 保存模型权重文件的间隔

# 打印信息的配置
log_config = dict(
    interval=100,   # 打印频率
    hooks=[ 
        # 打印信息的类型
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'