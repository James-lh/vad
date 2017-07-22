# encoding: utf-8

'''

@author: ZiqiLiu


@file: attention_config.py

@time: 2017/5/18 上午11:18

@desc:
'''


def get_config():
    return Config()


class Config(object):
    def __init__(self):
        self.mode = "train"  # train,valid,build
        self.ktq = False
        self.mfcc = False
        self.pre_emphasis = False

        self.model_path = './params/dnn1/'
        self.save_path = './params/dnn1/'
        self.graph_path = './graph/mel/'
        self.graph_name = 'graph.pb'
        #
        self.train_path = '/ssd/liuziqi/vad/train/'
        self.valid_path = '/ssd/liuziqi/vad/valid/'
        self.noise_path = '/ssd/liuziqi/vad/noise/'
        self.model_name = 'latest.ckpt'
        # self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'
        # self.train_path = './data/train/'
        # self.valid_path = './data/valid/'
        # self.noise_path = './data/noise/'

        # training flags
        self.reset_global = 0
        self.batch_size = 32
        self.tfrecord_size = 32
        self.valid_steps = 320
        self.gpu = "0"
        self.learning_rate = 2e-3
        self.max_epoch = 200
        self.valid_step = 320
        self.lr_decay = 0.8
        self.decay_step = 20000
        self.use_relu = True
        self.optimizer = 'adam'  # adam sgd nesterov

        self.fft_size = 400
        self.hop_size = 160
        self.samplerate = 16000
        self.max_sequence_length = 2000
        self.power = 1
        self.n_mfcc = 13
        self.n_mel = 26
        self.fmin = 300
        self.fmax = 8000
        self.validlen = 80000


        # noise flags
        self.use_white_noise = False
        self.bg_decay_max_db = -5
        self.bg_decay_min_db = 0
        self.bg_noise_prob = 0.5

        # model params
        self.num_layers = 2
        self.context_len = 5
        self.max_grad_norm = -1
        self.keep_prob = 0.9
        self.hidden_size = 128
        self.num_classes = 1
        self.res = False

        # valid params
        self.thres = 0.5

    @property
    def freq_size(self):
        return self.n_mfcc * 3 if self.mfcc else self.n_mel

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
