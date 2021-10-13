### Installation

* mmcv
* mmpose
* mmdetection
```
pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose/
pip3 install -r requirements.txt
pip3 install -v -e .  # or "python setup.py develop"
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip3 install -r requirements/build.txt
pip3 install -v -e .  # or "python setup.py develop"
cd ..
```
