#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
from SegmentationModel import PanopticSemanticSegModel
from batch_generator import BatchGenerator
from config import PanopticConfig
import matplotlib.pyplot as plt
import cv2
import pickle


def semantic_inference():
    """
    Train phase main process
    :return:
    """
    config = PanopticConfig()
    if not os.path.exists(config.dump_root_dir):
        os.makedirs(config.dump_root_dir)

    batch_generator = BatchGenerator(config=config)
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    with tf.compat.v1.Session(config=session_config) as sess:
        model = PanopticSemanticSegModel(config=config)

        tf.compat.v1.global_variables_initializer().run()
        saver = tf.compat.v1.train.Saver(max_to_keep=None)
        saver.restore(sess,
                      os.path.join(
                          config.dump_model_para_root_dir,
                          'epoch_13_train_loss_0.214400_val_loss_0.210000.ckpt')
                      )

        # global inference phase
        batch_index = 0
        while True:
            image_batch, gt_batch, semantic_gt_batch, imgs_names_batch, valid = \
                batch_generator.next_batch_inference()
            if not valid:
                break

            val_batch_loss, semantic_seg_probs = sess.run(
                fetches=[
                    model.loss,
                    model.semantic_seg_probs,
                ],
                feed_dict={
                    model.input_data: image_batch,
                    model.ground_truth: gt_batch
                })

            batch_index += 1

            print('[Global Inference] Batch {}: loss = {}.'.format(
                batch_index,
                np.round(val_batch_loss, 4)
            ))

            # dump into local disk
            dump_to_local_disk(
                config.dump_root_dir,
                imgs_names_batch,
                image_batch,
                semantic_gt_batch,
                semantic_seg_probs)

        sess.close()


def dump_to_local_disk(
        dump_root_dir,
        imgs_names_batch,
        image_batch,
        semantic_gt_batch,
        semantic_seg_probs):
    for i in range(len(imgs_names_batch)):
        names = imgs_names_batch[i].split('/')
        resized_input_image_full_path = os.path.join(
            dump_root_dir, '{}_{}_{}'.format(names[-3], names[-2], names[-1]))
        semantic_gt_full_path = os.path.join(
            dump_root_dir, '{}_{}_{}'.format(names[-3], names[-2], names[-1].split('.')[0] + '_semantic_gt.jpg'))
        semantic_infer_pkl_full_path = os.path.join(
            dump_root_dir, '{}_{}_{}'.format(names[-3], names[-2], names[-1].split('.')[0] + '_semantic_infer.pkl'))
        semantic_infer_full_path = os.path.join(
            dump_root_dir, '{}_{}_{}'.format(names[-3], names[-2], names[-1].split('.')[0] + '_semantic_infer.jpg'))

        mapping = [0.0, 1.0, 161.0, 33.0, 35.0, 36.0, 162.0, 38.0, 39.0, 165.0, 34.0, 40.0,
                   164.0, 37.0, 167.0, 168.0, 49.0, 50.0, 163.0, 65.0, 66.0, 67.0, 81.0, 82.0, 83.0,
                   84.0, 85.0, 86.0, 97.0, 99.0, 100.0, 113.0, 255.0]

        predicted_frame = np.argmax(semantic_seg_probs[i], axis=-1)
        predicted_result = np.zeros(shape=(predicted_frame.shape[0], predicted_frame.shape[1]))
        for col in np.arange(0, predicted_frame.shape[0]):                  # width
            for row in np.arange(0, predicted_frame.shape[1]):              # height
                predicted_result[col][row] = mapping[predicted_frame[col][row]]

        cv2.imwrite(resized_input_image_full_path, image_batch[i])
        cv2.imwrite(semantic_gt_full_path, semantic_gt_batch[i])
        cv2.imwrite(semantic_infer_full_path, predicted_result)

        # for small size storage, here convert float to UINT8 type
        probs = semantic_seg_probs[i]
        probs *= 255.0
        probs = probs.astype(np.uint8)
        pickle.dump(probs, open(semantic_infer_pkl_full_path, 'wb'))


if __name__ == '__main__':
    semantic_inference()
