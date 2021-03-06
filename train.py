# encoding: utf-8

'''

@author: ZiqiLiu


@file: train.py

@time: 2017/5/18 上午11:03

@desc:
'''

# -*- coding:utf-8 -*-
# !/usr/bin/python

import os
import sys
import time
import pickle
import signal
import traceback
from args import parse_args
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.python.framework import graph_util
from models import dnn, rnn, attention
from reader import read_dataset
from utils.common import check_dir, path_join
from utils.prediction import frame_accurcacy, posterior_predict

from utils.wer import WERCalculator

DEBUG = False


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.wer_cal = WERCalculator([0, -1])

    def run(self, TrainingModel):

        graph = tf.Graph()
        with graph.as_default(), tf.Session() as sess:

            self.data = read_dataset(self.config)

            if config.mode == 'train':
                print('building training model....')
                with tf.variable_scope("model"):
                    self.train_model = TrainingModel(self.config,
                                                     self.data.batch_input_queue(),
                                                     is_train=True)
                    self.train_model.config.show()
                print('building valid model....')
                with tf.variable_scope("model", reuse=True):
                    self.valid_model = TrainingModel(self.config,
                                                     self.data.valid_queue(),
                                                     is_train=False)
            else:
                with tf.variable_scope("model", reuse=False):
                    self.valid_model = TrainingModel(self.config,
                                                     self.data.valid_queue(),
                                                     is_train=False)
            saver = tf.train.Saver()

            # restore from stored models
            files = glob(path_join(self.config.model_path, '*.ckpt.*'))

            if len(files) > 0:

                saver.restore(sess, path_join(self.config.model_path,
                                              self.config.model_name))
                print(('Model restored from:' + self.config.model_path))
            else:
                print("Model doesn't exist.\nInitializing........")
                sess.run(tf.global_variables_initializer())

            sess.run(tf.local_variables_initializer())
            tf.Graph.finalize(graph)

            best_accuracy = 1
            accu_loss = 0
            st_time = time.time()
            epoch_step = config.tfrecord_size * self.data.train_file_size // config.batch_size
            if os.path.exists(path_join(self.config.save_path, 'best.pkl')):
                with open(path_join(self.config.save_path, 'best.pkl'),
                          'rb') as f:
                    best_accuracy = pickle.load(f)
                    print('best accuracy', best_accuracy)
            else:
                print('best not exist')

            check_dir(self.config.save_path)

            if self.config.mode == 'train':

                if self.config.reset_global:
                    sess.run(self.train_model.reset_global_step)

                def handler_stop_signals(signum, frame):
                    global run
                    run = False
                    if not DEBUG:
                        print(
                            'training shut down, total setp %s, the model will be save in %s' % (
                                step, self.config.save_path))
                        saver.save(sess, save_path=(
                            path_join(self.config.save_path, 'latest.ckpt')))
                        print('best accuracy', best_accuracy)
                    sys.exit(0)

                signal.signal(signal.SIGINT, handler_stop_signals)
                signal.signal(signal.SIGTERM, handler_stop_signals)

                best_list = []
                best_threshold = 0.1
                best_count = 0
                # (miss,false,step,best_count)

                last_time = time.time()

                try:
                    sess.run([self.data.noise_stage_op,
                              self.data.noise_filequeue_enqueue_op,
                              self.train_model.stage_op,
                              self.train_model.input_filequeue_enqueue_op,
                              self.valid_model.stage_op,
                              self.valid_model.input_filequeue_enqueue_op])

                    va = tf.trainable_variables()
                    for i in va:
                        print(i.name)
                    while self.epoch < self.config.max_epoch:
                        _, _, _, _, _, l, lr, step, grads, vs = sess.run(
                            [self.train_model.train_op,
                             self.data.noise_stage_op,
                             self.data.noise_filequeue_enqueue_op,
                             self.train_model.stage_op,
                             self.train_model.input_filequeue_enqueue_op,
                             self.train_model.loss,
                             self.train_model.learning_rate,
                             self.train_model.global_step,
                             self.train_model.grads,
                             self.train_model.vs
                             ])
                        # print('-'*30)
                        # for i in vs:
                        #     print(i.sum())
                        # print('grads',len(grads))
                        # print('='*30)
                        # for g in grads:
                        #     print(g.sum())
                        epoch = step // epoch_step
                        accu_loss += l
                        if epoch > self.epoch:
                            self.epoch = epoch
                            print('accumulated loss', accu_loss)
                            saver.save(sess, save_path=(
                                path_join(self.config.save_path,
                                          'latest.ckpt')))
                            print('latest.ckpt save in %s' % (
                                path_join(self.config.save_path,
                                          'latest.ckpt')))
                            accu_loss = 0
                        if step % config.valid_step == 2:
                            print('epoch time ', (time.time() - last_time) / 60)
                            last_time = time.time()

                            total_speech = 0
                            total_silence = 0
                            total_miss = 0
                            total_false = 0
                            valid_batch = self.data.valid_file_size * config.tfrecord_size // config.batch_size
                            text = ""
                            for i in range(valid_batch):
                                soft, labels, seqlen, _, _ = sess.run(
                                    [
                                        self.valid_model.softmax,
                                        self.valid_model.labels,
                                        self.valid_model.seqLengths,
                                        self.valid_model.stage_op,
                                        self.valid_model.input_filequeue_enqueue_op])
                                np.set_printoptions(precision=4,
                                                    threshold=np.inf,
                                                    suppress=True)
                                # print('-----------------')
                                #
                                # print(labels[0])
                                # print(names[0].decode())
                                # print(self.valid_set[i * config.batch_size])
                                # for i in names:
                                #     print(i.decode())
                                # print(softmax.shape)
                                # for i, j in zip(soft, labels):
                                #     print(np.concatenate((i, j), 1))
                                #
                                # for i in range(len(soft)):
                                #     soft[i][seqlen[i]:] = 0

                                logits = posterior_predict(soft, config.thres)
                                target_speech, target_silence, miss, false_trigger = frame_accurcacy(
                                    logits, labels, seqlen)
                                total_speech += target_speech
                                total_silence += target_silence
                                total_miss += miss
                                total_false += false_trigger

                            miss_rate = round(total_miss / total_speech, 4)
                            false_rate = round(total_false / total_silence, 4)
                            print('--------------------------------')
                            print('epoch %d' % self.epoch)
                            print('training loss:' + str(l))
                            print('learning rate:', lr, 'global step', step)
                            print('miss rate:' + str(miss_rate))
                            print('false rate: ' + str(false_rate))

                            if miss_rate + false_trigger < best_accuracy:
                                best_accuracy = miss_rate + false_trigger
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.save_path,
                                               'best.ckpt')))
                                with open(path_join(
                                        self.config.save_path, 'best.pkl'),
                                        'wb') as f:
                                    best_tuple = (miss_rate, false_rate)
                                    pickle.dump(best_tuple, f)
                            if miss_rate + false_trigger < best_threshold:
                                best_count += 1
                                print('best_count', best_count)
                                best_list.append(
                                    (miss_rate, false_trigger, step,
                                     best_count))
                                saver.save(sess,
                                           save_path=(path_join(
                                               self.config.save_path,
                                               'best' + str(
                                                   best_count) + '.ckpt')))

                    print(
                        'training finished, total epoch %d, the model will be save in %s' % (
                            self.epoch, self.config.save_path))
                    saver.save(sess, save_path=(
                        path_join(self.config.save_path, 'latest.ckpt')))
                    print('best accuracy:%f' % (best_accuracy))

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                finally:
                    with open('best_list.pkl', 'wb') as f:
                        pickle.dump(best_list, f)
                    print('total time:%f hours' % (
                        (time.time() - st_time) / 3600))
                    # When done, ask the threads to stop.

            else:
                total = 0
                wrong = 0

                valid_batch = self.data.valid_file_size * config.tfrecord_size // config.batch_size

                for i in range(valid_batch):
                    # if i > 7:
                    #     break
                    ind = 14
                    logits, labels, _, _ = sess.run(
                        [self.valid_model.outputs,
                         self.valid_model.labels,
                         self.valid_model.stage_op,
                         self.valid_model.input_filequeue_enqueue_op])
                    np.set_printoptions(precision=4,
                                        threshold=np.inf,
                                        suppress=True)

                    total_count, wrong_count = frame_accurcacy(
                        logits, labels)
                    total += total_count
                    wrong += wrong_count

                accruracy = 1 - wrong / wrong

                # miss_rate = miss_count / target_count
                # false_accept_rate = false_count / total_count
                print('--------------------------------')
                print('accurcay: %f' % (accruracy))

    def build_graph(self, DeployModel):
        check_dir(self.config.graph_path)
        config_path = path_join(self.config.graph_path, 'config.pkl')
        graph_path = path_join(self.config.graph_path, self.config.graph_name)
        import pickle
        pickle.dump(self.config, open(config_path, 'wb'))

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as session:
            with tf.variable_scope("model"):
                model = DeployModel(config=config)

            print('Graph build finished')
            variable_names = [n.name for n in
                              tf.get_default_graph().as_graph_def().node]
            for n in variable_names:
                print(n)

            saver = tf.train.Saver()
            saver.restore(session, save_path=path_join(self.config.model_path,
                                                       'latest.ckpt'))
            print("model restored from %s" % config.model_path)

            frozen_graph_def = graph_util.convert_variables_to_constants(
                session, session.graph.as_graph_def(),
                ['model/inputX', 'model/rnn_initial_states',
                 'model/rnn_states', 'model/softmax'])
            tf.train.write_graph(
                frozen_graph_def,
                os.path.dirname(graph_path),
                os.path.basename(graph_path),
                as_text=False,
            )
            try:
                tf.import_graph_def(frozen_graph_def, name="")
            except Exception as e:
                print("!!!!Import graph meet error: ", e)
                exit()
            print('graph saved in %s' % graph_path)


if __name__ == '__main__':

    flags, model = parse_args()
    print(flags)
    if model == 'dnn':
        from config.dnn_config import get_config

        config = get_config()
        TrainingModel = dnn.DNN
        DeployModel = dnn.DeployModel
    elif model == 'rnn':
        from config.rnn_config import get_config

        config = get_config()
        TrainingModel = rnn.GRU
        DeployModel = rnn.DeployModel
    elif model == 'attention':
        from config.attention_config import get_config

        config = get_config()
        TrainingModel = attention.Attention
        DeployModel = attention.DeployModel
    else:
        raise Exception('model %s not defined!' % model)
    for key in flags:
        if flags[key] is not None:
            if not hasattr(config, key):
                print("WARNING: Invalid override with attribute %s" % (key))
            else:
                setattr(config, key, flags[key])

    runner = Runner(config)
    if config.mode == 'build':
        runner.build_graph(DeployModel)
    else:
        runner.run(TrainingModel)
