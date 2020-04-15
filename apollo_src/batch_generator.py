#!/usr/bin/env python
# -*-coding:utf-8-*-
import os
import cv2
import numpy as np
import json
import random
from config import PanopticConfig
random.seed(777)


class BatchGenerator(object):
    """
    batch generator for updated data set (the annotations are refinement)
    """
    def __init__(self, config):
        self._config = config

        if not os.path.exists(self._config.cache_root_dir):
            os.makedirs(self._config.cache_root_dir)

        self._samples_json_full_path = os.path.join(self._config.cache_root_dir, 'samples.json')
        self._labels_mapping_json_full_path = os.path.join(self._config.cache_root_dir, 'labels_mapping.json')

        # self.collect_samples()

        self._all_samples = json.load(open(self._samples_json_full_path, 'r'))
        self._all_samples_sorted = sorted(self._all_samples, key=lambda x: x['image_full_path'])

        labels = json.load(open(self._labels_mapping_json_full_path, 'r'))['labels_mapping']
        self._labels_mapping = self.get_one_hot_mapping(labels)
        self._train_samples, self._validation_samples = self.split_train_val_samples()
        self._train_batch_amount = len(self._train_samples) // self._config.batch_size
        self._val_batch_amount = len(self._validation_samples) // self._config.batch_size

        print('# of total samples = {}, training batches amount = {}, validation batches amount = {}'.format(
            len(self._all_samples),
            self._train_batch_amount,
            self._val_batch_amount
        ))

        self._train_batch_index, self._val_batch_index, self._infer_batch_index = 0, 0, 0

    @staticmethod
    def get_one_hot_mapping(labels):
        mapping = dict()
        for i, label in enumerate(labels):
            vector = np.zeros(shape=(len(labels),))
            vector[i] = 1.0
            mapping[label] = vector
        return mapping

    def collect_samples(self):
        all_samples = list()
        labels_mapping = list()
        index = 0

        train_val_images_root_dir = os.path.join(self._config.train_val_root_dir, 'ColorImage')
        records = [record for record in os.listdir(train_val_images_root_dir) if 'Record' in record]
        for record in records:
            record_root_dir = os.path.join(train_val_images_root_dir, record)
            cameras = [camera for camera in os.listdir(record_root_dir) if 'Camera' in camera]
            for camera in cameras:
                camera_root_dir = os.path.join(record_root_dir, camera)
                image_names = [name for name in os.listdir(camera_root_dir) if name.endswith('.jpg')]
                for frame_name in image_names:
                    index += 1
                    image_full_path = os.path.join(camera_root_dir, frame_name)
                    semantic_label_prefix = os.path.join(self._config.train_val_root_dir, 'Label', record, camera)
                    label_full_path = os.path.join(semantic_label_prefix, frame_name.split('.jpg')[0] + '_bin.png')

                    if not os.path.exists(label_full_path):
                        continue

                    # load annotated mask
                    mask = cv2.imread(label_full_path, cv2.IMREAD_GRAYSCALE)

                    if np.sum(mask) == 0:   # invalid semantic annotations
                        continue
                    else:
                        all_samples.append(
                            dict(
                                image_full_path=image_full_path,
                                label_full_path=label_full_path
                            )
                        )

                        labels_mapping.extend(list(np.unique(mask).astype(np.float64)))
                        labels_mapping = list(set(labels_mapping))

                        print('Process {}, index = {}, len(labels_mapping) = {}, labels_mapping = {}'.format(
                            label_full_path,
                            index,
                            len(labels_mapping),
                            labels_mapping
                        ))
        json.dump(
            all_samples,
            open(self._samples_json_full_path, 'w'),
            ensure_ascii=False,
            indent=4)

        json.dump(
            dict(labels_mapping=labels_mapping),
            open(self._labels_mapping_json_full_path, 'w'),
            ensure_ascii=False,
            indent=4)

    def split_train_val_samples(self, train_val_ratio=6.0):
        random.shuffle(self._all_samples)
        amount = len(self._all_samples)
        validation_samples_amount = amount - int(amount * train_val_ratio / (1.0 + train_val_ratio))
        validation_samples_amount = self._config.batch_size * (validation_samples_amount // self._config.batch_size)
        return self._all_samples[:-validation_samples_amount], self._all_samples[-validation_samples_amount:]

    def onehot_mapping(self, resized_mask):
        """
        convert ground truth of mask into one-hot vector format

        :param resized_mask: input ground truth batch
        :return:
        """
        # (width, height, channels)
        onehot_mask = np.zeros(shape=(resized_mask.shape[0], resized_mask.shape[1], len(self._labels_mapping)))
        for col in np.arange(0, resized_mask.shape[0]):         # width
            for row in np.arange(0, resized_mask.shape[1]):     # height
                onehot_mask[col][row] = self._labels_mapping[resized_mask[col][row]]

        return onehot_mask

    def next_train_batch(self):
        image_batch = list()
        gt_batch = list()
        for i in np.arange(self._train_batch_index * self._config.batch_size,
                           (self._train_batch_index + 1) * self._config.batch_size):
            # if achieves end of training samples, then random select sample from trianing samples
            if i >= len(self._train_samples):
                index = random.randint(0, len(self._train_samples)-1)
            else:
                index = i

            sample = self._train_samples[index]
            image_full_path = sample['image_full_path']
            label_full_path = sample['label_full_path']

            image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(label_full_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (self._config.width, self._config.height))
            resized_mask = cv2.resize(mask, (self._config.width, self._config.height), interpolation=cv2.INTER_NEAREST)

            resized_mask = self.onehot_mapping(resized_mask)

            image_batch.append(resized_image)
            gt_batch.append(resized_mask)

        self._train_batch_index += 1

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)
        return image_batch, gt_batch

    def next_val_batch(self):
        image_batch = list()
        gt_batch = list()
        for index in np.arange(self._val_batch_index * self._config.batch_size,
                               (self._val_batch_index + 1) * self._config.batch_size):
            sample = self._validation_samples[index]
            image_full_path = sample['image_full_path']
            label_full_path = sample['label_full_path']

            image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(label_full_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (self._config.width, self._config.height))
            resized_mask = cv2.resize(mask, (self._config.width, self._config.height), interpolation=cv2.INTER_NEAREST)

            resized_mask = self.onehot_mapping(resized_mask)

            image_batch.append(resized_image)
            gt_batch.append(resized_mask)

        self._val_batch_index += 1

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)
        return image_batch, gt_batch

    def next_batch_inference(self):
        image_batch = list()
        gt_batch = list()
        semantic_gt_batch = list()
        imgs_names_batch = list()

        valid = True

        # self._all_samples_sorted
        for index in np.arange(self._infer_batch_index * self._config.batch_size,
                               (self._infer_batch_index + 1) * self._config.batch_size):

            if (self._infer_batch_index + 1) * self._config.batch_size >= len(self._all_samples_sorted):
                valid = False
                break

            sample = self._all_samples_sorted[index]
            image_full_path = sample['image_full_path']
            label_full_path = sample['label_full_path']

            image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(label_full_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (self._config.width, self._config.height))
            semantic_mask = cv2.resize(mask, (self._config.width, self._config.height), interpolation=cv2.INTER_NEAREST)
            resized_mask = self.onehot_mapping(semantic_mask)

            image_batch.append(resized_image)
            semantic_gt_batch.append(semantic_mask)
            gt_batch.append(resized_mask)
            imgs_names_batch.append(image_full_path)

        self._infer_batch_index += 1

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)
        semantic_gt_batch = np.array(semantic_gt_batch)
        return image_batch, gt_batch, semantic_gt_batch, imgs_names_batch, valid

    @property
    def train_batch_amount(self):
        return self._train_batch_amount

    @property
    def val_batch_amount(self):
        return self._val_batch_amount

    @property
    def config(self):
        return self._config

    def reset_validation_batches(self):
        self._val_batch_index = 0

    def reset_training_batches(self):
        random.shuffle(self._train_samples)
        self._train_batch_index = 0


if __name__ == '__main__':
    batch_generator = BatchGenerator(
        config=PanopticConfig()
    )
