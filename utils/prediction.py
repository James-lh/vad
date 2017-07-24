# encoding: utf-8

'''

@author: ZiqiLiu


@file: prediction.py

@time: 2017/6/14 下午4:30

@desc:
'''

import numpy as np


def posterior_predict(probs, thres):
    return np.where(probs > thres, 1, 0)


def frame_accurcacy(logits, labels, seqlen):
    # shape=(b,t,1)
    assert logits.shape == labels.shape
    target_speech = 0
    target_silence = 0
    miss = 0
    false_trigger = 0
    logits = _smoothing(logits)
    logits = np.squeeze(logits, 2)
    labels = np.squeeze(labels, 2)
    for logit, label, seql in zip(logits, labels, seqlen):
        lo = logit[:seql]
        la = label[:seql]
        target_speech += lo.sum()
        target_silence += seql - lo.sum()
        for i, j in zip(lo, la):
            if i == 0 and j == 1:
                false_trigger += 1
            elif i == 1 and j == 0:
                miss += 1

    # for i,j in zip(logits,labels):
    #     print(i)
    #     print(j)
    #     print('+'*20)
    # print(target_speech, target_silence, miss, false_trigger)
    return target_speech, target_silence, miss, false_trigger


def _smoothing(logits, step=2):
    accu = np.copy(logits)
    for i in range(1, step + 1):
        shift_before = np.pad(logits[:, 1:, :], ((0, 0), (0, 1), (0, 0)),
                              mode='constant',
                              constant_values=0)
        shift_after = np.pad(logits[:, :-1, :], ((0, 0), (1, 0), (0, 0)),
                             mode='constant',
                             constant_values=0)
        accu += shift_after
        accu += shift_before
    result = np.where(accu > step + 1, 1, 0)
    return result
