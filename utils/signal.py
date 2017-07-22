# encoding: utf-8

'''

@author: ZiqiLiu


@file: signal.py

@time: 2017/7/20 下午4:52

@desc:
'''
import numpy as np


def compute_db(signal):
    energy = np.square(signal).sum() / len(signal)
    db = 10 * np.log10(energy)
    return db
