FROM pytorch/pytorch

COPY .. server/

WORKDIR ./server


# python
# pip install 
RUN pip install -r ../requirement.txt -i https://pypi.douban.com/simple

# /val/output
CMD python3 ../tools/train.py  ../config/body/2d_kpt_single_view_rgb_img/deeppose/coco/res_coco_256x192.py