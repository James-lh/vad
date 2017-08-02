# encoding: utf-8

'''

@author: ZiqiLiu


@file: attention_ctc.py

@time: 2017/5/18 上午11:04

@desc:
'''

# !/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
import librosa
import math
from utils.common import describe
from utils.shape import tf_frame, window
from utils.mfcc import mfcc
from positional_encoding import positional_encoding_op


def self_attention(inputs, config, is_training, scope_name='self_attention'):
    # inputs - [batch_size, time_steps, model_size]
    # return - [batch_size, time_steps, model_size]
    assert (config.hidden_size % config.multi_head_num == 0)
    head_size = config.hidden_size // config.multi_head_num
    with tf.variable_scope(scope_name):
        combined = tf.layers.conv2d(
            tf.expand_dims(inputs, 2), 3 * config.hidden_size,
            (1, 1), name="qkv_transform")
        q, k, v = tf.split(
            tf.squeeze(combined, 2),
            [config.hidden_size, config.hidden_size, config.hidden_size],
            axis=2)

        q_ = tf.concat(tf.split(q, config.multi_head_num, axis=2),
                       axis=0)  # (h * B, T, N / h)
        k_ = tf.concat(tf.split(k, config.multi_head_num, axis=2),
                       axis=0)  # (h * B, T, N / h)
        v_ = tf.concat(tf.split(v, config.multi_head_num, axis=2),
                       axis=0)  # (h * B, T, N / h)

        a = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # (h * B, T, T)
        a = tf.nn.softmax(a / math.sqrt(head_size))
        if is_training:
            a = tf.nn.dropout(a, config.keep_prob)
        a = tf.matmul(a, v_)  # [h * B, T, N / h]

        outputs = tf.concat(tf.split(a, config.multi_head_num, axis=0),
                            axis=2)  # [B, T, N]

    return outputs


def feed_forward(inputs, config, scope_name='feed_forward'):
    # inputs - [batch_size, time_steps, model_size]
    # return - [batch_size, time_steps, model_size]
    with tf.variable_scope(scope_name):
        inners = tf.layers.conv2d(
            tf.expand_dims(inputs, 2), config.feed_forward_inner_size,
            (1, 1), activation=tf.nn.relu, name="conv1")  # [B, T, 1, F]
        outputs = tf.layers.conv2d(
            inners, config.hidden_size, (1, 1), name="conv2")  # [B, T, 1, F]
    return tf.squeeze(outputs, 2)


def inference(inputs, seqLengths, config, is_training, batch_size=None):
    if not batch_size:
        batch_size = config.batch_size
    # positional encoding
    max_length = tf.shape(inputs)[1]
    if config.combine_frame > 1:
        padding = tf.zeros([1, 1, config.freq_size])
        padding = tf.tile(padding,
                          [batch_size,
                           config.combine_frame - tf.mod(max_length,
                                                         config.combine_frame),
                           1])
        inputs = tf.concat([inputs, padding], 1)
        inputs = tf.reshape(inputs, [batch_size, -1,
                                     config.freq_size * config.combine_frame])
        seqLengths = seqLengths // config.combine_frame + 1
        max_length = max_length // config.combine_frame + 1

    inputs = tf.layers.conv2d(
        tf.expand_dims(inputs, 2), config.hidden_size, (1, 1),
        name='input_linear_trans')  # [B, T, 1, F]
    inputs = tf.squeeze(inputs, 2)  # [B, T, F]

    pe = positional_encoding_op.positional_encoding(
        max_length, config.hidden_size)
    inputs = inputs + pe

    layer_inputs = inputs
    for j in range(config.num_layers):
        with tf.variable_scope('layer_%d' % j):
            # self attention sub-layer
            attention_outputs = self_attention(layer_inputs, config,
                                               is_training)
            if is_training:
                attention_outputs = tf.nn.dropout(
                    attention_outputs, config.keep_prob)
            # add and norm
            feed_forward_inputs = tf.contrib.layers.layer_norm(
                attention_outputs + layer_inputs)
            # feed forward sub-layer
            feed_forward_outputs = feed_forward(feed_forward_inputs, config)
            if is_training:
                feed_forward_outputs = tf.nn.dropout(
                    feed_forward_outputs, config.keep_prob)
            # add and norm
            layer_outputs = tf.contrib.layers.layer_norm(
                feed_forward_outputs + feed_forward_inputs)
            layer_inputs = layer_outputs

    outputs = tf.layers.conv2d(
        tf.expand_dims(layer_outputs, 2), config.num_classes,
        (1, 1), name='output_linear_trans')  # [B, T, 1, F]
    outputs = tf.squeeze(outputs, 2)  # [B, T, F]
    if config.use_relu:
        outputs = tf.nn.relu(outputs)
    return outputs, seqLengths


class Attention(object):
    def __init__(self, config, input, is_train):
        self.config = config

        stager, self.stage_op, self.input_filequeue_enqueue_op = input

        self.inputX, self.labels, self.seqLengths = stager.get()
        if is_train:
            self.inputX = tf.nn.dropout(self.inputX, config.keep_prob)
        self.build_graph(config, is_train)

    @describe
    def build_graph(self, config, is_train):

        self.nn_inputs = window(self.inputX, config.context_len, 1)

        self.nn_outputs, self.seqLengths = inference(self.nn_inputs,
                                                     self.seqLengths, config,
                                                     is_train)

        if is_train:
            self.labels = tf.cast(self.labels, tf.float32)
            self.xent_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.nn_outputs, name='xent')
            self.loss = tf.reduce_mean(self.xent_loss, name='loss')
            self.global_step = tf.Variable(0, trainable=False)
            self.reset_global_step = tf.assign(self.global_step, 1)

            initial_learning_rate = tf.Variable(
                config.learning_rate, trainable=False)

            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate, self.global_step, self.config.decay_step,
                self.config.lr_decay, name='lr')

            if config.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif config.optimizer == 'nesterov':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                            0.9,
                                                            use_nesterov=True)
            else:
                raise Exception('optimizer not defined')

            self.vs = tf.trainable_variables()
            grads_and_vars = self.optimizer.compute_gradients(self.loss,
                                                              self.vs)
            self.grads = [grad for (grad, var) in grads_and_vars]
            self.vs = [var for (grad, var) in grads_and_vars]
            if config.max_grad_norm > 0:
                self.grads, _ = tf.clip_by_global_norm(
                    self.grads, config.max_grad_norm)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.vs),
                global_step=self.global_step)
        else:
            self.labels = tf.slice(self.labels, [0, 0, 1], [-1, -1, -1])

            self.softmax = tf.nn.softmax(self.nn_outputs, name='softmax')
            self.softmax = tf.slice(self.softmax, [0, 0, 1], [-1, -1, -1])


class DeployModel(object):
    def __init__(self, config):
        """
        In deployment, we use placeholder to get input. Only inference part
        are built. seq_lengths, rnn_states, rnn_outputs, ctc_decode_inputs
        are exposed for streaming decoding. All operators are placed in CPU.
        Padding should be done before data is fed.
        """
        config.keep_prob = 1

        self.inputX = tf.placeholder(dtype=tf.float32,
                                     shape=[None, ],
                                     name='inputX')
        self.inputX = tf.expand_dims(self.inputX, 0)
        self.frames = tf_frame(self.inputX, 400, 160, name='frame')

        self.linearspec = tf.abs(tf.spectral.rfft(self.frames, [400]))

        if config.mfcc:
            self.melspec = mfcc(self.linearspec, config, batch_size=1)
        else:
            self.mel_basis = librosa.filters.mel(
                sr=config.samplerate,
                n_fft=config.fft_size,
                fmin=config.fmin,
                fmax=config.fmax,
                n_mels=config.freq_size).T
            self.mel_basis = tf.constant(value=self.mel_basis, dtype=tf.float32)
            self.mel_basis = tf.expand_dims(self.mel_basis, 0)

            self.melspec = tf.matmul(self.linearspec, self.mel_basis,
                                     name='mel')

        # self.melspec = tf.expand_dims(self.melspec, 0)

        self.fuck = tf.identity(self.melspec, name='fuck')

        self.seqLengths = tf.expand_dims(tf.shape(self.melspec)[1], 0)
        self.nn_outputs, self.new_seqLengths = inference(self.melspec,
                                                         self.seqLengths,
                                                         config,
                                                         is_training=False,
                                                         batch_size=1)

        self.softmax = tf.nn.softmax(self.nn_outputs, name='softmax')
        self.softmax = tf.slice(self.softmax, [0, 0, 1], [-1, -1, -1],
                                name='output_prob')


if __name__ == "__main__":
    pass
