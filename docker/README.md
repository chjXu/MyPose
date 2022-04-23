# Docker 打包教程

# 1. 准备项目文件夹
```
 |── config
 |── data
 |── demo
 ...
```

# 2. 编写Dockerfile文件
```dockerfile
FROM pytorch/pytorch    # 指定运行环境

WORKDIR /mypose         # 指定工作目录

COPY .. mypose          # 拷贝指令

# python
# pip install 
RUN pip install -r requirement.txt -i https://pypi.douban.com/simple

# /val/output
CMD python3 ../demo/top_down_img_demo.py ../config/body/2d_kpt_single_view_rgb_img/deeppose/coco/res_coco_256x192.py ../tools/work_dirs/res_coco_256x192/best_AP_epoch_31.pth --img-root ../tests/data/coco --json-file ../tests/data/coco/test_coco.json --out-img-root vis_result

# /train
CMD python3 ../tools/train.py  ../config/body/2d_kpt_single_view_rgb_img/deeppose/coco/res_coco_256x192.py
```

# 3. 在有Dockerfile的文件下运行，创建镜像
```
    docker build -t example:latest .
```
> -t 镜像名 .表示当前路径下的所有文件

# 4. 查看当前创建的镜像文件
```
    docker images
```

查看是否存在镜像文件（example）

# (Options) 保存镜像文件到本地
```
    docker save example > /path/to/you/want/example.tar
```

# 5. 上传至Docker Hub

## 5.1 登录Docker hub

```
    sudo docker login
```

成功的话，会输出提示信息。在此之前，应该注册一个Docker hub账号

## 5.2 将本地Docker镜像的tag更改为正确tag格式
```
    sudo docker tag example YOUR_DOCKER_ID/example
```

## 5.3 上传
```
    sudo docker push YOUR_DOCKER_ID/example
```

# 6. 打开Docker Hub桌面版

运行：

```
    docker run -d -p 80:80 YOUR_DOCKER_ID/example
```

你可以在本地仓库里看到这个镜像了

# 7. 点击run即可运行


# 可能遇到的bug
1. 权限问题
> 建议放在用户目录下创建

2. libGL.so文件不存在
> 从网上查到，因为opencv-python库只能在本地运行，若想在docker中运行，需要卸载对应的opencv-python，安装对应的opencv-python-headless库即可


# 常用指令

docker --version     # 查看版本

