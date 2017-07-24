# encoding: utf-8

'''

@author: ZiqiLiu


@file: make_valid.py

@time: 2017/7/20 下午4:41

@desc:
'''
import pickle
from utils.common import path_join,check_dir
from utils.signal import compute_db
import librosa
import random

db_decay = -5
noise_file = '/ssd/keyword/noise/babble.wav'

from config.dnn_config import get_config

config = get_config()

wave_valid_dir = config.rawdata_path + 'train/'
save_valid_dir = config.rawdata_path + 'valid/vad/'
check_dir(save_valid_dir)


def generate(path):
    noise, _ = librosa.load(noise_file)

    with open(path, 'rb') as f:
        wav_list = pickle.load(f)
    print('read pkl from %s' % f)
    # each record should be (file.wav,((st,end),(st,end).....)))
    file_list = [i[0] for i in wav_list]
    for f in file_list:
        y, _ = librosa.load(path_join(wave_valid_dir, f), sr=config.samplerate)
        wave_db = compute_db(y)
        start = random.randint(0, len(noise) - len(y))
        temp = noise[start:start + len(y)]

        noise_db = compute_db(temp)
        factor = wave_db - noise_db + db_decay
        factor = 10 ** (factor / 10)
        temp *= factor
        y += temp
        librosa.output.write_wav(path_join(save_valid_dir, f), y,
                                 sr=config.samplerate)


generate(path_join(config.rawdata_path,'valid/vad_valid.pkl'))