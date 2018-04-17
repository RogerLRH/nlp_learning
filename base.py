# -*- coding: utf-8 -*-
import pickle

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
    return [w for w in words if w not in stopwords]


def save_dict(dic, dictfile):
    pickle.dump(dic, open(dictfile, "wb"))


def get_reverse_dict(dic):
    redic = {}
    for word, idx in dic.items():
        redic[idx] = word
    return redic


class Text2index(object):
    def __init__(self):
        self.dict = {}

    def load_dict(self, dictfile):
        self.dict = pickle.load(open(dictfile, "rb"))

    def text2index(self, text, dense=True):
        # 词列表转化为index列表
        res = [self.dict.get(word, 0) for word in text]
        if dense:
            return [i for i in res if i]
        return res


def pad_sentence(x, max_len, forward=False):
    if forward:
        new = (x + [0] * max_len)[:max_len]
    else:
        new = ([0] * max_len + x)[-max_len:]
    return np.array(new, dtype=int)


def pad_array(x, size, forward=False):
    m, n = size[0], size[1]
    if forward:
        x = x[:m]
    else:
        x = x[-m:]
    x = np.array([pad_sentence(i, n, forward=forward) for i in x], dtype=int)
    if len(x) != m:
        x = np.concatenate((np.zeros((m - len(x), n), dtype=int)), x)
    return x


def get_max_size(inputs):
    m, n, dim = 0, 0, 1
    if isinstance(inputs[0][0], list):
        dim = 2
    for item in inputs:
        if len(item) > m:
            m = len(item)
        if dim == 2 and len(item[0]) > n:
            n = len(item[0])
    return [m, n]


class TextIndexLoader(object):
    def __init__(self, dataset, batch_size, fix_size=None, shuffle=True, forward=False):
        """
        dataset: 数据集，两种情况：1.只含有输入；2.大小为2的tuple，分别对应输入和标签
        fix_size: 整数或长度为2的整数列表，指定输入的大小，None或0代表输入大小随batch改变
        """
        if len(dataset) == 2:
            self.inputs = dataset[0]
            self.labels = dataset[1]
        else:
            self.inputs = dataset
            self.labels = None
        self.batch_size = batch_size
        if fix_size and isinstance(fix_size, int):
            fix_size = [fix_size, 0]
        self.fix_size = fix_size
        self.shuffle = shuffle
        self.forward = forward
        self.num = len(self.inputs)
        self.num_batch = (self.num - 1) // self.batch_size + 1

    def __iter__(self):
        return self.loader()

    def __len__(self):
        return self.num

    def loader(self):
        sf = np.array(range(self.num))
        if self.shuffle:
            np.random.shuffle(sf)
        for i in range(self.num_batch):
            idx = sf[self.batch_size*i:self.batch_size*(i+1)]
            inputs = [self.inputs[i] for i in idx]
            if not self.fix_size:
                size = get_max_size(inputs)
            else:
                size = self.fix_size
            if size[1]:
                inputs = np.array([pad_array(i, size, forward=self.forward) for i in inputs], dtype=int)
            else:
                inputs = np.array([pad_sentence(i, size[0], forward=self.forward) for i in inputs], dtype=int)
            if self.labels is not None:
                labels = np.array([self.labels[i] for i in idx], dtype=int)
                yield inputs, labels
            else:
                yield inputs
