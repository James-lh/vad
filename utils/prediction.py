# encoding: utf-8

'''

@author: ZiqiLiu


@file: prediction.py

@time: 2017/6/14 下午4:30

@desc:
'''

import numpy as np


def frame_accurcacy(logits, labels):
    # shape=(b,t)
    assert logits.shape == labels.shape
    logits = _smoothing(logits)
    xor = logits ^ labels
    return xor.size, xor.sum


def _smoothing(logits, step=2):
    accu = np.copy(logits)
    for i in range(1, step + 1):
        shift_before = np.pad(logits[1:], ((0, 0), (0, 1)), mode='constant',
                              constant_values=0)
        shift_after = np.pad(logits[:-1], ((0, 0), (1, 0)), mode='constant',
                             constant_values=0)
        accu += shift_after
        accu += shift_before
    result = np.where(accu > step + 1, 1, 0)
    return result
