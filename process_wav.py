# encoding: utf-8

'''

@author: ZiqiLiu


@file: process_wav.py

@time: 2017/5/19 下午2:28

@desc:
'''
import librosa
import numpy as np
from config.dnn_config import get_config
import pickle
import tensorflow as tf
from utils.common import check_dir, path_join, increment_id

config = get_config()

wave_train_dir = config.rawdata_path + 'train/'
wave_valid_dir = config.rawdata_path + 'valid/vad/'
wave_noise_dir = config.rawdata_path + 'noise/'

save_train_dir = config.train_path
save_valid_dir = config.valid_path
save_noise_dir = config.noise_path

global_len = []
temp_list = []
error_list = []

validlen = config.validlen


def pre_emphasis(signal, coefficient=0.97):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def time2frame(second, sr=config.samplerate, n_fft=config.fft_size,
               hop_size=config.hop_size):
    return int((second * sr) / hop_size)


def point2frame(point, step_size=config.hop_size):
    return point / step_size


def convert_label(times, seq_len):
    label = np.zeros([seq_len, 1], dtype=np.int32)
    for start, end in times:
        fr_start = time2frame(start)
        fr_end = time2frame(end)
        label[fr_start:fr_end + 1] = 1
    return label


def process_stft(f):
    y, sr = librosa.load(f, sr=config.samplerate)
    if config.pre_emphasis:
        y = pre_emphasis(y)
    linearspec = np.transpose(np.abs(
        librosa.core.stft(y, config.fft_size,
                          config.hop_size)))


    return linearspec, y


def process_mel(f):
    y, sr = librosa.load(f, sr=config.samplerate)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size,
                                       hop_length=config.hop_size,
                                       power=2.,
                                       fmin=300,
                                       fmax=8000,
                                       n_mels=config.num_features))

    return mel_spectrogram, y


def make_record(f, label):
    # print(f)
    # print(text)
    spectrogram, wave = process_stft(f)
    seq_len = spectrogram.shape[0]
    label = convert_label(label, seq_len)

    return spectrogram, seq_len, label


def make_example(spectrogram, seq_len, label):
    spectrogram = spectrogram.tolist()
    label = label.tolist()
    ex = tf.train.SequenceExample()

    ex.context.feature["seq_len"].int64_list.value.append(seq_len)

    fl_audio = ex.feature_lists.feature_list["audio"]
    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)

    int_label = ex.feature_lists.feature_list['label']
    for frame in label:
        int_label.feature.add().int64_list.value.extend(frame)
    return ex


def make_noise_example(spectrogram):
    spectrogram = spectrogram.tolist()
    ex = tf.train.SequenceExample()
    ex.context.feature["seq_len"].int64_list.value.append(
        config.max_sequence_length)
    fl_audio = ex.feature_lists.feature_list["audio"]

    for frame in spectrogram:
        fl_audio.feature.add().float_list.value.extend(frame)

    return ex


def batch_padding(tup_list):
    # tuple : (spec,labels,seqlen)
    new_list = []
    max_len = max([len(t[0]) for t in tup_list])

    for t in tup_list:
        assert (len(t[0]) == len(t[1]))
        paded_wave = np.pad(t[0], pad_width=(
            (0, max_len - t[0].shape[0]), (0, 0)),
                            mode='constant', constant_values=0)
        paded_label = np.pad(t[1], pad_width=(
            (0, max_len - t[0].shape[0]), (0, 0)),
                             mode='constant', constant_values=0)

        new_list.append((paded_wave, paded_label, t[2]))
    return new_list


def batch_padding_valid(tup_list):
    # tuple : (spec,labels,seqlen)
    new_list = []

    for t in tup_list:
        padlen = validlen - len(t[0])
        pad_left = padlen // 2
        pad_right = (padlen + 1) // 2
        paded_wave = np.pad(t[0], pad_width=(
            (pad_left, pad_right), (0, 0)),
                            mode='constant', constant_values=0)
        paded_label = np.pad(t[1], pad_width=(
            (pad_left, pad_right), (0, 0)),
                             mode='constant', constant_values=0)

        new_list.append((paded_wave, paded_label, t[2]))
    return new_list


def generate_trainning_data(path):
    with open(path, 'rb') as f:
        wav_list = pickle.load(f)
    print('read pkl from %s' % f)
    # each record should be (file.wav,((st,end),(st,end).....)))
    file_list = [i[0] for i in wav_list]
    label_list = [i[1] for i in wav_list]
    tuple_list = []
    counter = 0
    record_count = 0
    for i, audio_name in enumerate(file_list):
        spec, seq_len, labels = make_record(
            path_join(wave_train_dir, audio_name),
            label_list[i])
        counter += 1

        tuple_list.append((spec, labels, seq_len))
        if counter == config.tfrecord_size:
            tuple_list = batch_padding(tuple_list)
            fname = 'data' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [make_example(spec, seq_len, labels) for
                       spec, labels, seq_len in tuple_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(save_train_dir, fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            tuple_list.clear()
            print(fname, 'created')
    print('save in %s' % save_train_dir)


def generate_valid_data(path):
    with open(path, 'rb') as f:
        wav_list = pickle.load(f)
    print('read pkl from %s' % f)
    # each record should be (file.wav,((st,end),(st,end).....)))
    file_list = [i[0] for i in wav_list]
    label_list = [i[1] for i in wav_list]
    tuple_list = []
    counter = 0
    record_count = 0
    for i, audio_name in enumerate(file_list):
        spec, seq_len, labels = make_record(
            path_join(wave_valid_dir, audio_name),
            label_list[i])

        counter += 1
        tuple_list.append((spec, labels, seq_len))
        if counter == config.tfrecord_size:
            tuple_list = batch_padding(tuple_list)
            fname = 'data' + increment_id(record_count, 5) + '.tfrecords'
            ex_list = [make_example(spec, seq_len, labels) for
                       spec, labels, seq_len in tuple_list]
            writer = tf.python_io.TFRecordWriter(
                path_join(save_valid_dir, fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            counter = 0
            tuple_list.clear()
            print(fname, 'created')
    print('save in %s' % save_valid_dir)


def generate_noise_data(path):
    with open(path, 'rb') as f:
        audio_list = pickle.load(f)
        print('read pkl from ', f)
    spec_list = []
    record_count = 0
    for i, audio_name in enumerate(audio_list):
        spec, y = process_stft(path_join(wave_noise_dir, audio_name))
        if spec.shape[0] >= config.max_sequence_length:
            spec_list.extend(
                split_spectrogram(spec, config.max_sequence_length))
        else:
            spec_list.append(
                expand_spectrogram(spec, config.max_sequence_length))

        if len(spec_list) >= config.tfrecord_size:

            fname = 'noise' + increment_id(record_count, 5) + '.tfrecords'
            temp = spec_list[:config.tfrecord_size]
            spec_list = spec_list[config.tfrecord_size:]
            ex_list = [make_noise_example(spec) for spec in temp]
            writer = tf.python_io.TFRecordWriter(
                path_join(save_noise_dir, fname))
            for ex in ex_list:
                writer.write(ex.SerializeToString())
            writer.close()
            record_count += 1
            print(fname, 'created')
    print('save in %s' % save_noise_dir)


def split_spectrogram(spec, target_len):
    result = []
    for i in range(0, spec.shape[0] - target_len, target_len):
        result.append(spec[i:i + target_len])
    return result


def expand_spectrogram(spec, target_len):
    times = target_len // spec.shape[0]
    expand_spec = spec
    for i in range(times):
        expand_spec = np.concatenate((expand_spec, spec), 0)
    return expand_spec[:target_len]


def sort_wave(pkl_path):
    def get_len(f):
        y, sr = librosa.load(f, sr=config.samplerate)
        return len(y)

    import re
    dir = re.sub(r'[^//]+.pkl', '', pkl_path)

    with open(pkl_path, "rb") as f:
        training_data = pickle.load(f)
        sorted_data = sorted(training_data,
                             key=lambda a: get_len(dir + a[0]))
    with open(pkl_path + '.sorted', "wb") as f:
        pickle.dump(sorted_data, f)

    y, sr = librosa.load(dir + sorted_data[-1][0])
    print(len(y))


if __name__ == '__main__':
    check_dir(save_train_dir)
    check_dir(save_valid_dir)
    check_dir(save_noise_dir)

    base_pkl = 'vad_train.pkl'
    # sort_wave(wave_train_dir + base_pkl)
    # generate_trainning_data(
    #     wave_train_dir + base_pkl + '.sorted')

    # sort_wave(wave_valid_dir + "vad_valid.pkl")
    # generate_valid_data(wave_valid_dir + "vad_valid.pkl.sorted")

    generate_noise_data(wave_noise_dir + 'vad_noise.pkl')
