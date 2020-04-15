#!/usr/bin/env python
import tensorflow as tf


class PanopticSemanticSegModel(object):
    """
    image semantic segmentation
    """
    def __init__(self, config):
        """
        configure model parameters and build model
        :param config: model configuration parameters
        """
        self._config = config
        self._image_width = config.width
        self._image_height = config.height
        self._learning_rate = config.learning_rate
        self._batch_size = config.batch_size
        self._channels = 3      # BGR channels

        # foreground and background, should be defined as
        # the total categories amount of semantic segmentation
        self._num_cls = 33

        self._input_data = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                self._batch_size,
                self._image_height,
                self._image_width,
                self._channels],
            name="input_image")

        self._ground_truth = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                self._batch_size,
                self._image_height,
                self._image_width,
                self._num_cls],
            name="ground_truth")

        self._global_step = tf.Variable(0, trainable=False)
        self._optimizer, self._summary_op = None, None
        self._all_trainable_vars = None
        self._semantic_seg_probs = None
        self._semantic_seg_logits = None
        self._loss = None

        self.build_model()

    def build_model(self):
        """
        build Unet model
        :return:
        """
        def conv2d(filter_shape, strides, padding='SAME', activation=tf.nn.relu):
            """
            conv2d wrapper
            :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
            :param strides: specifying the strides of the convolution along the height and width
            :param padding:
            :param activation:
            :return:
            """
            conv = tf.keras.layers.Conv2D(
                filters=filter_shape[-1],
                kernel_size=filter_shape[0:2],
                strides=strides,
                padding=padding,
                activation=activation,
                )
            return conv

        def upconv2d(filter_shape, strides, padding='SAME'):
            """
            conv2d transpose wrapper
            :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
            :param strides: A list of 2 positive integers specifying the strides of the convolution
            :param padding:
            :return:
            """
            upconv = tf.keras.layers.Conv2DTranspose(
                filters=filter_shape[-1],
                kernel_size=filter_shape[0:2],
                strides=strides,
                padding=padding,
                activation=tf.nn.relu,
            )
            return upconv

        with tf.compat.v1.variable_scope("U-Net-Backbone"):
            # net down
            image = self._input_data
            conv1_1 = conv2d(filter_shape=[3, 3, 3, 16], strides=[1, 1])(image)
            conv1_2 = conv2d(filter_shape=[3, 3, 16, 16], strides=[1, 1])(conv1_1)
            pool_1 = tf.nn.max_pool2d(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv2_1 = conv2d(filter_shape=[3, 3, 16, 32], strides=[1, 1])(pool_1)
            conv2_2 = conv2d(filter_shape=[3, 3, 32, 32], strides=[1, 1])(conv2_1)
            pool_2 = tf.nn.max_pool2d(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3_1 = conv2d(filter_shape=[3, 3, 32, 48], strides=[1, 1])(pool_2)
            conv3_2 = conv2d(filter_shape=[3, 3, 48, 48], strides=[1, 1])(conv3_1)
            pool_3 = tf.nn.max_pool2d(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv4_1 = conv2d(filter_shape=[3, 3, 48, 64], strides=[1, 1])(pool_3)
            conv4_2 = conv2d(filter_shape=[3, 3, 64, 64], strides=[1, 1])(conv4_1)
            pool_4 = tf.nn.max_pool2d(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # bottom
            conv5_1 = conv2d(filter_shape=[3, 3, 64, 64], strides=[1, 1])(pool_4)
            conv5_2 = conv2d(filter_shape=[3, 3, 64, 64], strides=[1, 1])(conv5_1)

            # net up
            upconv6_1 = upconv2d(filter_shape=[3, 3, 64, 32], strides=[2, 2])(conv5_2)
            concat6_1 = tf.concat([upconv6_1, conv4_2], axis=3)
            conv6_2 = conv2d(filter_shape=[3, 3, 96, 64], strides=[1, 1])(concat6_1)
            conv6_3 = conv2d(filter_shape=[3, 3, 64, 64], strides=[1, 1])(conv6_2)

            upconv7_1 = upconv2d(filter_shape=[3, 3, 64, 32], strides=[2, 2])(conv6_3)
            concat7_1 = tf.concat([upconv7_1, conv3_2], axis=3)
            conv7_2 = conv2d(filter_shape=[3, 3, 80, 48], strides=[1, 1])(concat7_1)
            conv7_3 = conv2d(filter_shape=[3, 3, 48, 48], strides=[1, 1])(conv7_2)

            upconv8_1 = upconv2d(filter_shape=[3, 3, 48, 32], strides=[2, 2])(conv7_3)
            concat8_1 = tf.concat([upconv8_1, conv2_2], axis=3)
            conv8_2 = conv2d(filter_shape=[3, 3, 64, 32], strides=[1, 1])(concat8_1)
            conv8_3 = conv2d(filter_shape=[3, 3, 32, 32], strides=[1, 1])(conv8_2)

            upconv9_1 = upconv2d(filter_shape=[3, 3, 32, 32], strides=[2, 2])(conv8_3)
            concat9_1 = tf.concat([upconv9_1, conv1_2], axis=3)
            conv9_2 = conv2d(filter_shape=[3, 3, 48, 48], strides=[1, 1])(concat9_1)
            conv9_3 = conv2d(filter_shape=[3, 3, 48, 48], strides=[1, 1])(conv9_2)

        with tf.compat.v1.variable_scope("SemanticSegmentation"):
            self._semantic_seg_logits = conv2d(filter_shape=[1, 1, 48, self._num_cls], strides=[1, 1])(conv9_3)
            self._semantic_seg_probs = tf.nn.softmax(logits=self._semantic_seg_logits)

        with tf.compat.v1.variable_scope("SemanticSegmentationLoss"):
            flatten_ground_truth = tf.reshape(self._ground_truth, [-1, self._num_cls])
            flatten_semantic_seg_logits = tf.reshape(self._semantic_seg_logits, [-1, self._num_cls])

            flatten_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=flatten_ground_truth,
                logits=flatten_semantic_seg_logits,
            )

            self._loss = tf.reduce_mean(flatten_loss)
            tf.compat.v1.summary.scalar('loss', self._loss)

        with tf.compat.v1.variable_scope("params-stat"):
            self._all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])

        with tf.compat.v1.variable_scope("optimization"):
            train_op = tf.compat.v1.train.AdamOptimizer(self._learning_rate)
            self._optimizer = train_op.minimize(self._loss)
            self._summary_op = tf.compat.v1.summary.merge_all()

    @property
    def loss(self):
        return self._loss

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def input_data(self):
        return self._input_data

    @property
    def semantic_seg_logits(self):
        return self._semantic_seg_logits

    @property
    def semantic_seg_probs(self):
        return self._semantic_seg_probs

    @property
    def loss(self):
        return self._loss

    @property
    def all_trainable_vars(self):
        return self._all_trainable_vars
