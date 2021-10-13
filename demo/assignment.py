# Copyright (c) OpenMMLab. All rights reserved.
#new_new
import os
import warnings
from argparse import ArgumentParser
import threading
import concurrent.futures
from typing import Match
import numpy as np
from PIL import Image
import shutil
import os

import cv2

blackblankimage = 255 * np.zeros((1080,1920,3), np.uint8)


def getFrame(cap):
    flag, img = cap.read()
    if(flag == 0):
        img = blackblankimage.copy()
    return flag,img


def FrameCapture(path, vid):
      
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
  
    while success:
        try:
            success, image = vidObj.read()
            vid.append(image)
            count += 1
        except:
            break







from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    vid1 = [] 
    vid2 = [] 
    vid3 = [] 
    vid4 = []

    t1 = threading.Thread(target=FrameCapture, args=("/content/drive/MyDrive/1_.mp4",vid1) , name = 't1')
    t2 = threading.Thread(target=FrameCapture, args=("/content/drive/MyDrive/2_.mp4",vid2), name = 't2')
    t3 = threading.Thread(target=FrameCapture, args=("/content/drive/MyDrive/3_.mp4",vid3), name = 't3')
    t4 = threading.Thread(target=FrameCapture, args=("/content/drive/MyDrive/4_.mp4",vid4), name = 't4')

    t1.start(), t2.start(), t3.start(), t4.start()
    t1.join(), t2.join(), t3.join(), t4.join()

    l = [len(vid1), len(vid2), len(vid3), len(vid4)]
    mx = max(l)

    blackblankimage = 255 * np.zeros((1080,1920,3), np.uint8)


    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # cap = cv2.VideoCapture(args.video_path)
    # cap = cv2.VideoCapture("/content/drive/MyDrive/1_.mp4") 
    fps = None

    # assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 24.0
        size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []

    cap1 = cv2.VideoCapture("/content/drive/MyDrive/1__.mp4")
    cap2 = cv2.VideoCapture("/content/drive/MyDrive/2__.mp4")
    cap3 = cv2.VideoCapture("/content/drive/MyDrive/3__.mp4")
    cap4 = cv2.VideoCapture("/content/drive/MyDrive/4__.mp4")

    caps = [cap1, cap2, cap3, cap4]

    while (cap1.isOpened() or cap2.isOpened() or cap3.isOpened() or cap4.isOpened()):
        pose_results_last = pose_results

        with concurrent.futures.ThreadPoolExecutor() as executor:
            t1 = executor.submit(getFrame, cap1)
            t2 = executor.submit(getFrame, cap2)
            t3 = executor.submit(getFrame, cap3)
            t4 = executor.submit(getFrame, cap4)
            
            flag1, img1 = t1.result()
            flag2, img2 = t2.result()
            flag3, img3 = t3.result()
            flag4, img4 = t4.result()

            if(flag1 == 0 and flag2 == 0 and flag3 == 0 and flag4 == 0):
                break


        img12 = cv2.hconcat([img1, img2])
        img34 = cv2.hconcat([img3, img4])
        img = cv2.vconcat([img12, img34])
        img = cv2.resize(img, (1920, 1080))

        # flag, img = cap.read()
        # if not flag:
        #     break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=fps)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # if save_out_video:
    #     videoWriter.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


