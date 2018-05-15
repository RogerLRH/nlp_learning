# -*- coding: utf-8 -*-
import numpy as np


def is_chinese(chara):
    if u'\u4e00' <= chara <= u'\u9fa5':
        return True
    return False


def is_english(chara):
    if u'\u0061' <= chara <= u'\u007a' or u'\u0041' <= chara <= u'\u005a':
        return True
    return False


def is_number(chara):
    if u'\u0030' <= chara <= u'\u0039':
        return True
    return False


def label2onehot(labels, num_class):
    return np.eye(num_class)[labels]


def load_lines2list(filename):
    res = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.strip())
    return res


def remove_stopword(words, stopwords):
    """remove stopwords in words.

    Parameters
    ----------
    words : string list
    stopwords : list or set

    Returns
    -------
    string list
        words without stopwords
    """
    stopwords = set(stopwords)
    return [w for w in words if w not in stopwords]
