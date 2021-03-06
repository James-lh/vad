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
        self.mfcc = False
        self.ktq = False

        self.model_path = './params/attention/'
        self.save_path = './params/attention/'
        self.graph_path = './graph/23w/'
        self.graph_name = 'graph.pb'

        self.train_path = '/ssd/liuziqi/ctc_23w/train/'
        self.valid_path = '/ssd/liuziqi/ctc_23w/valid/'
        self.noise_path = '/ssd/liuziqi/ctc_23w/noise/'
        self.model_name = 'latest.ckpt'
        self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'


        # training flags
        self.reset_global = 0
        self.batch_size = 16
        self.tfrecord_size = 32
        self.valid_steps = 320
        self.gpu = "0"
        self.warmup = False
        self.learning_rate = 1e-3
        self.max_epoch = 200
        self.valid_step = 320
        self.lr_decay = 0.9
        self.decay_step = 40000
        self.use_relu = True
        self.optimizer = 'adam'  # adam sgd nesterov

        # pre process flags
        self.fft_size = 400
        self.hop_size = 160
        self.samplerate = 16000
        self.max_sequence_length = 2000
        self.power = 1
        self.fmin = 300
        self.fmax = 8000
        self.n_mfcc = 20
        self.n_mel = 60
        self.pre_emphasis = False

        # noise flags
        self.use_white_noise = False
        self.bg_decay_max_db = -5
        self.bg_decay_min_db = 5
        self.bg_noise_prob = 0.8

        # model params
        self.combine_frame = 2
        self.num_layers = 3
        self.max_grad_norm = -1
        self.feed_forward_inner_size = 512
        self.keep_prob = 0.9
        self.multi_head_num = 8
        self.hidden_size = 128
        self.num_classes = 1

    @property
    def freq_size(self):
        return self.n_mfcc * 3 if self.mfcc else self.n_mel

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
