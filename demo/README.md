# 1. 以gt框为输入的测试
```python
    python ./top_down_img_demo.py ../config/body/2d_kpt_single_view_rgb_img/deeppose/coco/res_coco_256x192.py ../tools/work_dirs/res_coco_256x192/deeppose_res50_coco_256x192.pth --img-root ../tests/data/coco/ --json-file ../tests/data/coco/test_coco.json --out-img-root vis_result --device=cpu
```

# 2. 使用目标检测器 (图像)
```python
    python ./top_down_img_demo_with_det.py ./mydetection_cfg/faster_rcnn_r50_fpn_coco.py ../tools/work_dirs/res_coco_256x192/faster_rcnn_r50_coco.pth ../config/body/2d_kpt_single_view_rgb_img/deeppose/coco/res_coco_256x192.py ../tools/work_dirs/res_coco_256x192/deeppose_res50_coco_256x192.pth --img-root ../tests/data/coco/ --img 000000001000.jpg --out-img-root vis_result --device=cpu
```

# 3. 使用目标检测器 (视频)
```python
python ./top_down_video_demo_with_det.py ./mydetection_cfg/faster_rcnn_r50_fpn_coco.py ../tools/work_dirs/res_coco_256x192/faster_rcnn_r50_coco.pth ../config/body/2d_kpt_single_view_rgb_img/deeppose/coco/res_coco_256x192.py ../tools/work_dirs/res_coco_256x192/deeppose_res50_coco_256x192.pth --video-path ./demo.mp4 --out-video-root vis_results
```