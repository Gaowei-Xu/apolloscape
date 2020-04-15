#!/usr/bin/python
import os
import cv2
import numpy as np


class ImageTools(object):
    def __init__(self, root_dir, video_root_dir):
        self._root_dir = root_dir
        self._video_root_dir = video_root_dir

        if not os.path.exists(self._video_root_dir):
            os.makedirs(self._video_root_dir)

    def run(self):
        image_names = [name for name in os.listdir(self._root_dir) if name.endswith('.jpg')]
        panoptic_seg_names = [name for name in image_names if 'panoptic' in name]
        semantic_seg_names = [name for name in image_names if '_semantic_infer' in name]
        instances_seg_names = [name for name in image_names if '_instance_infer' in name]
        input_img_names = [name for name in image_names if 'infer' not in name and 'gt' not in name]
        input_img_names = sorted(input_img_names)
        panoptic_seg_names = sorted(panoptic_seg_names)
        semantic_seg_names = sorted(semantic_seg_names)
        instances_seg_names = sorted(instances_seg_names)

        fps = 12
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            os.path.join(self._video_root_dir, 'combine_demo.avi'),
            fourcc,
            fps,
            (960, 640))

        frames_amount = min(len(panoptic_seg_names), len(semantic_seg_names), len(instances_seg_names))
        for i in range(frames_amount):
            # print(input_img_names[i])
            # print(panoptic_seg_names[i])
            # print(semantic_seg_names[i])
            # print(instances_seg_names[i])

            input_image = cv2.imread(os.path.join(self._root_dir, input_img_names[i]), cv2.IMREAD_COLOR)
            semantic_seg = cv2.imread(os.path.join(self._root_dir, semantic_seg_names[i]), cv2.IMREAD_COLOR)
            instances_seg = cv2.imread(os.path.join(self._root_dir, instances_seg_names[i]), cv2.IMREAD_COLOR)
            panoptic_seg = cv2.imread(os.path.join(self._root_dir, panoptic_seg_names[i]), cv2.IMREAD_COLOR)
            combine_frame_1 = np.concatenate([input_image, semantic_seg], axis=1)
            combine_frame_2 = np.concatenate([instances_seg, panoptic_seg], axis=1)
            combine_frame = np.concatenate([combine_frame_1, combine_frame_2], axis=0)
            video_writer.write(combine_frame)
            print('Processing frame index {}/{}...'.format(i+1, frames_amount))
        video_writer.release()


if __name__ == '__main__':
    image_tools = ImageTools(
        root_dir='../dump/',
        video_root_dir='../demo_videos/'
    )

    image_tools.run()

