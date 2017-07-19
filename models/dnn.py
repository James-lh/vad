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
from utils.common import describe
from utils.shape import tf_frame
from utils.mfcc import mfcc
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def window(inputs, context_length, frame_step):
    # inputs (b,t,h)
    # return (b,t,5h)
    batch_size, frame_length, features = inputs.get_shape().as_list()
    frame_length = tf.shape(inputs)[1]

    window_len = context_length * 2 + 1
    padding = tf.pad(inputs, ((0, 0), (context_length, context_length), (0, 0)),
                     mode="REFLECT")

    indices_frame = array_ops.expand_dims(math_ops.range(window_len), 0)
    indices_frames = array_ops.tile(indices_frame, [frame_length, 1])

    indices_step = array_ops.expand_dims(
        math_ops.range(frame_length) * frame_step, 1)
    indices_steps = array_ops.tile(indices_step, [1, window_len])

    indices = indices_frames + indices_steps

    padding = array_ops.transpose(padding, [1, 2, 0])
    window_frames = array_ops.gather(padding, indices)
    window_frames = array_ops.transpose(window_frames, perm=[3, 0, 1, 2])
    output = tf.reshape(window_frames,
                        [batch_size, frame_length, window_len * features])
    return output


def linear_transform(inputs, batch_size, input_size, output_size,
                     weights_name='dnn_weights', bias_name='dnn_bias'):
    flatten_inputs = tf.reshape(inputs, [-1, input_size])
    weights = tf.get_variable(weights_name, [input_size, output_size],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(bias_name, [output_size])

    outputs = tf.reshape((tf.matmul(flatten_inputs, weights) + bias),
                         [batch_size, -1, output_size])
    return outputs


def inference(inputs, seqLengths, config, is_training, batch_size=None,
              activation=tf.nn.relu):
    if not batch_size:
        batch_size = config.batch_size

    layer_inputs = linear_transform(inputs, batch_size,
                                    (config.context_len * 2 + 1) * config.n_mel,
                                    config.hidden_size, 'input_linear_weights',
                                    'input_linear_bias')

    for j in range(config.num_layers):
        with tf.variable_scope('layer_%d' % j):
            # self attention sub-layer
            layer_outputs = linear_transform(layer_inputs, batch_size,
                                             config.hidden_size,
                                             config.hidden_size)
            layer_outputs = activation(layer_outputs)
            if is_training:
                layer_outputs = tf.nn.dropout(
                    layer_outputs, config.keep_prob)
            # add and norm
            if config.res:
                layer_outputs = tf.contrib.layers.layer_norm(
                    layer_outputs + layer_inputs)
            layer_inputs = layer_outputs

    # fully connection to output
    nn_output = linear_transform(layer_outputs, batch_size, config.hidden_size,
                                 config.num_classes, 'fc_weights', 'fc_bias')
    if config.use_relu:
        nn_output = tf.nn.relu(nn_output)
    return nn_output


class DNN(object):
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

        self.nn_outputs = inference(self.nn_inputs, self.seqLengths, config,
                                    is_train)

        if is_train:
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.nn_outputs)
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
            self.softmax = tf.nn.softmax(self.nn_outputs, name='softmax')
            self.outputs = tf.where(tf.greater(self.softmax, config.thres),
                                    tf.ones_like(self.softmax),
                                    tf.zeros_like(self.softmax))


class DeployModel(object):
    def __init__(self, config):
        """
        In deployment, we use placeholder to get input. Only inference part
        are built. seq_lengths, rnn_states, rnn_outputs, ctc_decode_inputs
        are exposed for streaming decoding. All operators are placed in CPU.
        Padding should be done before data is fed.
        """

        # input place holder
        config.keep_prob = 1

        # with tf.device('/cpu:0'):
        #
        # self.inputX = tf.placeholder(dtype=tf.float32,
        #                              shape=[None, config.fft_size],
        #                              name='inputX')
        #
        # complex_tensor = tf.complex(
        #     self.inputX,
        #     imag=tf.zeros_like(self.inputX, dtype=tf.float32),
        #     name='complex_tensor')
        # abs = tf.abs(
        #     tf.fft(complex_tensor, name='fft'))
        # print(abs)

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


if __name__ == "__main__":
    pass
