### Installation

* mmcv
* mmpose
* mmdetection
```
#mmcv
pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

#mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose/
pip3 install -r requirements.txt
pip3 install -v -e .  # or "python setup.py develop"
cd ..

#mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip3 install -r requirements/build.txt
pip3 install -v -e .  # or "python setup.py develop"
cd ..
```

### Execution

First need to clone the updated codebase. You can find the orginal codebase is [here](https://github.com/open-mmlab/mmpose.git).
```
git clone https://github.com/qwertyz15/mmpose.git
```
Then we can able to run the program through this command.
```
cd mmpose/
python3 demo/assignment.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path demo/resources/passed.mp4 \
    --out-video-root task_output_video
cd ..
```
### Output
The output video of the program will be saved ``` task_output_video/vis_passed.mp4 ``` here.

Here are some snaps of the output video.

![alt text](https://github.com/qwertyz15/mmpose/blob/master/demo/resources/ss_4.png?raw=true)

![alt text](https://github.com/qwertyz15/mmpose/blob/master/demo/resources/ss_3.png?raw=true)

![alt text](https://github.com/qwertyz15/mmpose/blob/master/demo/resources/ss_2.png?raw=true)

![alt text](https://github.com/qwertyz15/mmpose/blob/master/demo/resources/ss_1.png?raw=true)

