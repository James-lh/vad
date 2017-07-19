# encoding: utf-8

"""

@author: ZiqiLiu


@file: reader.py

@time: 2017/5/11 下午12:33

@desc:
"""
import numpy as np
import tensorflow as tf
import math
from utils.common import path_join
import librosa
from glob import glob
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import random_ops
from utils.mfcc import mfcc


class DataSet(object):
    def __init__(self, config, train_dir, valid_dir, noise_dir, mode='train'):
        self.config = config
        # freq size of linear spectrogram
        self.last_dim = 1 + config.fft_size // 2

        self.mel_basis = librosa.filters.mel(
            sr=config.samplerate,
            n_fft=config.fft_size,
            fmin=config.fmin,
            fmax=config.fmax,
            n_mels=config.freq_size).T
        self.mel_basis = tf.constant(value=self.mel_basis, dtype=tf.float32)
        self.mel_basis = tf.tile(tf.expand_dims(self.mel_basis, 0),
                                 [config.batch_size, 1, 1])
        if mode == 'train':
            print(train_dir)
            self.train_filename = glob(path_join(train_dir, '*.tfrecords'))
            # self.train_filename = sorted(self.train_filename)
            self.train_file_size = len(self.train_filename)
            if self.train_file_size == 0:
                raise Exception('train tfrecords not found')
            self.train_reader = tf.TFRecordReader(name='train_reader')
            print('file size', self.train_file_size)

            self.noise_filename = glob(path_join(noise_dir, '*.tfrecords'))
            self.noise_reader = tf.TFRecordReader(name='noise_reader')
            if len(self.noise_filename) == 0:
                raise Exception(
                    'noise tfrecords not found. Or disable noise option')

        self.valid_filename = glob(path_join(valid_dir, '*.tfrecords'))
        # self.valid_filename = sorted(self.valid_filename)
        self.valid_file_size = len(self.valid_filename)
        if self.valid_file_size == 0:
            raise Exception('valid tfrecords not found')
        self.valid_reader = tf.TFRecordReader(name='valid_reader')
        self.validation_size = len(self.valid_filename) * config.tfrecord_size
        print('validation size', self.validation_size)

    def train_filequeue_reader(self, filename_queue):
        (keys, values) = self.train_reader.read_up_to(filename_queue,
                                                      self.config.batch_size)
        context_features = {
            "seq_len": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        seq_features = {
            "audio": tf.FixedLenSequenceFeature([self.last_dim],
                                                dtype=tf.float32),
            "label": tf.FixedLenSequenceFeature([1],
                                                dtype=tf.int64)
        }

        audio_list = []
        label_list = []
        len_list = []

        for i in range(self.config.batch_size):
            context, sequence = tf.parse_single_sequence_example(
                serialized=values[i],
                context_features=context_features,
                sequence_features=seq_features
            )
            audio = sequence['audio']
            label = sequence['label']
            seq_len = context['seq_len']

            audio_list.append(audio)
            label_list.append(label)
            len_list.append(seq_len)

        seq_lengths = tf.cast(tf.reshape(tf.stack(len_list, name='seq_lengths'),
                                         (-1,)), tf.int32)
        audio_tensor = tf.stack(audio_list, name='input_audio')
        label_tensor = tf.stack(label_list, name='input_label')

        return audio_tensor, label_tensor, seq_lengths

    def valid_filequeue_reader(self, filename_queue):
        (keys, values) = self.train_reader.read_up_to(filename_queue,
                                                      self.config.batch_size)
        context_features = {
            "seq_len": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        seq_features = {
            "audio": tf.FixedLenSequenceFeature([self.last_dim],
                                                dtype=tf.float32),
            "label": tf.FixedLenSequenceFeature([1],
                                                dtype=tf.int64)
        }

        audio_list = []
        label_list = []
        len_list = []

        for i in range(self.config.batch_size):
            context, sequence = tf.parse_single_sequence_example(
                serialized=values[i],
                context_features=context_features,
                sequence_features=seq_features
            )
            audio = sequence['audio']
            label = sequence['label']
            seq_len = context['seq_len']

            audio_list.append(audio)
            label_list.append(label)
            len_list.append(seq_len)

        seq_lengths = tf.cast(tf.reshape(tf.stack(len_list, name='seq_lengths'),
                                         (-1,)), tf.int32)
        audio_tensor = tf.stack(audio_list, name='input_audio')
        label_tensor = tf.stack(label_list, name='input_label')

        return audio_tensor, label_tensor, seq_lengths

    def noise_filequeue_reader(self, filename_queue):
        (keys, values) = self.noise_reader.read_up_to(filename_queue,
                                                      self.config.batch_size,
                                                      name='read_noise')
        audio_features = {
            "audio": tf.FixedLenSequenceFeature([self.last_dim],
                                                dtype=tf.float32)
        }
        audio_list = []
        len_list = []

        for i in range(self.config.batch_size):
            context, sequence = tf.parse_single_sequence_example(
                serialized=values[i],
                sequence_features=audio_features
            )
            audio = sequence['audio']
            seq_len = tf.shape(audio)[0]
            audio_list.append(audio)
            len_list.append(seq_len)

        seq_lengths = tf.stack(len_list, name='noise_lengths')

        return tf.stack(audio_list, name='noise_audio'), seq_lengths

    def string_input_queue(self, string_tensor, shuffle=True,
                           name=None, seed=None, capacity=16384):
        with ops.name_scope(name, "input_producer", [string_tensor]) as name:
            input_tensor = ops.convert_to_tensor(
                string_tensor, dtype=dtypes.string)
            if shuffle:
                input_tensor = random_ops.random_shuffle(input_tensor,
                                                         seed=seed)
            q = data_flow_ops.FIFOQueue(
                capacity=capacity,
                dtypes=[input_tensor.dtype.base_dtype])
            enq = tf.cond(tf.less(q.size(), 2),
                          lambda: q.enqueue_many([input_tensor]),
                          lambda: tf.no_op())
            return q, enq

    def batch_input_queue(self, shuffle=True):
        with tf.device('/cpu:0'):
            self.train_filename_queue, self.train_filequeue_enqueue_op = self.string_input_queue(
                self.train_filename, shuffle=shuffle, capacity=16384)
            linearspec, labels, seq_len = self.train_filequeue_reader(
                self.train_filename_queue)

            self.noise_stager, self.noise_stage_op, self.noise_filequeue_enqueue_op = self.noise_queue(
                True)

        bg_noise_origin, bg_noise_lengths = self.noise_stager.get()

        # all noise wave already padded to max_seq_len
        noise_length = self.config.max_sequence_length

        audio_db = self.compute_db(linearspec, seq_len)
        noise_db = self.compute_db(
            bg_noise_origin, bg_noise_lengths)
        decay = tf.random_uniform(
            [], self.config.bg_decay_min_db,
            self.config.bg_decay_max_db, name="bg_decay")
        factor = audio_db - noise_db + decay

        factor = 10 ** (factor / 20)
        factor = tf.tile(
            tf.reshape(factor, [-1, 1, 1]),
            [1, noise_length, self.last_dim])
        bg_noise = bg_noise_origin * factor

        spec_length = tf.shape(linearspec)[1]
        start = tf.random_uniform(
            [], minval=0, maxval=noise_length - spec_length,
            dtype=tf.int32, name="random_select_start")
        bg_noise_slice = tf.slice(
            bg_noise, [0, start, 0],
            [self.config.batch_size, spec_length, self.last_dim],
            name="slice_noise")

        add_bg_noise = tf.random_uniform([], 0, 1)
        linearspec = tf.cond(
            tf.less(add_bg_noise, self.config.bg_noise_prob),
            lambda: linearspec + bg_noise_slice,
            lambda: linearspec)

        if self.config.use_white_noise:
            print('use white noise')
            with tf.name_scope("white_noise"):
                noise = tf.random_uniform(
                    [self.config.batch_size, tf.shape(linearspec)[1],
                     self.last_dim],
                    minval=0, maxval=1e-3)
                linearspec = linearspec + noise

        if self.config.mfcc:
            melspec = mfcc(linearspec, self.config, None)
        else:
            if self.config.power == 2:
                linearspec = tf.square(linearspec)
            melspec = tf.matmul(linearspec, self.mel_basis)

        stager = data_flow_ops.StagingArea(
            [tf.float32, tf.int64, tf.int32],
            shapes=[(self.config.batch_size, None, self.config.freq_size),
                    (self.config.batch_size, None, 1),
                    (self.config.batch_size,)])

        stage_op = stager.put((melspec, labels, seq_len))

        return stager, stage_op, self.train_filequeue_enqueue_op

    def valid_queue(self):
        with tf.device('/cpu:0'):
            self.valid_filename_queue, self.valid_filequeue_enqueue_op = self.string_input_queue(
                self.valid_filename, shuffle=False, capacity=16384)
            linearspec, labels, seq_len = self.valid_filequeue_reader(
                self.valid_filename_queue)

        if self.config.mfcc:
            melspec = mfcc(linearspec, self.config, None)
        else:
            if self.config.power == 2:
                linearspec = tf.square(linearspec)
            melspec = tf.matmul(linearspec, self.mel_basis)

        stager = data_flow_ops.StagingArea(
            [tf.float32, tf.int64, tf.int32],
            shapes=[(self.config.batch_size, None, self.config.freq_size),
                    (self.config.batch_size, None, 1),
                    (self.config.batch_size,)])

        stage_op = stager.put(
            (melspec, labels, seq_len))

        return stager, stage_op, self.valid_filequeue_enqueue_op

    def noise_queue(self, shuffle=True):
        with tf.device('/cpu:0'):
            noise_filename_queue, noise_filequeue_enqueue_op = self.string_input_queue(
                self.noise_filename, shuffle=shuffle, capacity=16384)

            audio, seq_len = self.noise_filequeue_reader(
                noise_filename_queue)

        stager = data_flow_ops.StagingArea(
            [tf.float32, tf.int32],
            shapes=[
                (self.config.batch_size, None, self.last_dim),
                (self.config.batch_size)])

        stage_op = stager.put((audio, seq_len))

        return stager, stage_op, noise_filequeue_enqueue_op

    def compute_db(self, spectrum, lengths):
        with tf.name_scope("compute_db"):
            energy = tf.reduce_sum(tf.square(spectrum), [1, 2]) / \
                     self.config.fft_size
            energy = energy / self.config.fft_size / \
                     tf.cast(lengths, dtype=tf.float32)
            db = 10 * tf.log(energy) / math.log(10)
        return db


def read_dataset(config):
    save_train_dir = config.train_path
    save_valid_dir = config.valid_path
    save_noise_dir = config.noise_path

    return DataSet(config=config, train_dir=save_train_dir,
                   valid_dir=save_valid_dir, noise_dir=save_noise_dir,
                   mode=config.mode, )
