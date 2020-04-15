#!/usr/bin/env python


class PanopticConfig(object):
    def __init__(self):
        self._learning_rate = 0.001
        self._train_val_root_dir = '../dataset/road02_ins/'
        self._cache_root_dir = '../cache/'

        self._batch_size = 48

        self._width = 480
        self._height = 320

        self._max_epoch = 250
        self._train_summary_root_dir = '../apollo_train_logs/'
        self._dump_model_para_root_dir = '../apollo_model_params/'
        self._save_every_epoch = 1

        self._dump_root_dir = '../dump/'

    @property
    def train_val_root_dir(self):
        return self._train_val_root_dir

    @property
    def cache_root_dir(self):
        return self._cache_root_dir

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def train_summary_root_dir(self):
        return self._train_summary_root_dir

    @property
    def dump_model_para_root_dir(self):
        return self._dump_model_para_root_dir

    @property
    def save_every_epoch(self):
        return self._save_every_epoch

    @property
    def dump_root_dir(self):
        return self._dump_root_dir
