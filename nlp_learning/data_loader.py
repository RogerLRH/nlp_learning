# -*- coding: utf-8 -*-
import torch
import numpy as np


def pad_list(x, max_len, forward=False):
    """
    Padding an int list.

    Parameters
    ----------
    x : int list
    max_len : int
    forward : bool
        To get front part if True. Default: False.

    Returns
    -------
    int list
    """
    if forward:
        new = (x + [0] * max_len)[:max_len]
    else:
        new = ([0] * max_len + x)[-max_len:]
    return np.array(new, dtype=int)


def pad_array(x, size, forward=False):
    """
    Padding a list of int list.

    Parameters
    ----------
    x : list of int list
    size : tuple or list with two int
        array size to pad
    forward : bool
        To get front part if True. Default: False.

    Returns
    -------
    int array
    """
    m, n = size[0], size[1]
    if forward:
        x = x[:m]
    else:
        x = x[-m:]
    x = np.array([pad_list(i, n, forward=forward) for i in x], dtype=int)
    if len(x) != m:
        added = np.zeros((m - len(x), n), dtype=int)
        x = np.concatenate((added, x))
    return x


def pad_general(item, size, forward):
    if isinstance(size, list):
        item = [pad_array(i, size, forward=forward) for i in item]
    else:
        item = [pad_list(i, size, forward=forward) for i in item]
    return np.array(item, dtype=int)


def get_max_size(inputs):
    m, n, array = 0, 0, False
    if isinstance(inputs[0][0], list):
        array = True
    for item in inputs:
        if len(item) > m:
            m = len(item)
        if array and len(item[0]) > n:
            n = len(item[0])
    if array:
        return [m, n]
    return m


class ClassLoader(object):
    """
    An endless iteration which gives batches of indexed text datas.

    Parameters
    ----------
    inputs : list
        Each element should be same type(index list or list of index list), the length is free.
    labels : list
        Each element should be same type(int for single label or int list for multilabels), if list, the length should be same. Can be None for predict case. Default: None.
    input_size : int or list
        If given, each sample size is the same. If not, sample size is depend on the max size of batch's samples, which is variable. Default: None.
    batch_size : int
        Default: 128.
    shuffle : bool
        If shuffle the data at every epoch. Default: True.
    forward : bool
        When input_size is given, to get front part for padding if True. Default: False.
    """
    def __init__(self, inputs, labels=None, input_size=None, batch_size=128, shuffle=True, forward=False):
        self._inputs = inputs
        self._labels = labels
        self._input_size = input_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._forward = forward
        self._num = len(self._inputs)
        self._num_batch = (self._num - 1) // self._batch_size + 1

    def __iter__(self):
        return self._loader()

    def __len__(self):
        return self._num

    def _padding(self, inputs, size):
        return pad_general(inputs, size, self._forward)

    def _input_treator(self, idx):
        inputs = [self._inputs[i] for i in idx]
        if self._input_size is None:
            size = get_max_size(inputs)
        else:
            size = self._input_size
        return self._padding(inputs, size)

    def _label_treator(self, idx):
        return np.array([self._labels[i] for i in idx], dtype=int)

    def _loader(self):
        sf = np.array(range(self._num))
        if self._shuffle:
            np.random.shuffle(sf)
        for i in range(self._num_batch):
            idx = sf[self._batch_size*i:self._batch_size*(i+1)]
            inputs = self._input_treator(idx)
            if self._labels is not None:
                labels = self._label_treator(idx)
                yield inputs, labels
            else:
                yield inputs


class TorchClassLoader(ClassLoader):
    def _padding(self, inputs, size):
        inputs = pad_general(inputs, size, self._forward)
        return torch.LongTensor(inputs)

    def _label_treator(self, idx):
        labels = np.array([self._labels[i] for i in idx], dtype=int)
        return torch.LongTensor(labels)


class TranslationLoader(ClassLoader):
    """
    An endless iteration which gives batches of indexed text datas.

    Parameters
    ----------
    inputs : list
        Each element should be same type(index list), the length is free.
    labels : list
        Each element should be same type(int for single label or int list for multilabels), if list, the length should be same. Can be None for predict case. Default: None.
    input_size : int
        If given, each sample size is the same. If not, sample size is depend on the max size of batch's samples, which is variable. Default: None.
    label_size : int
        If given, each sample size is the same. If not, sample size is depend on the max size of batch's samples, which is variable. Default: None.
    batch_size : int
        Default: 128.
    shuffle : bool
        If shuffle the data at every epoch. Default: True.
    forward : bool
        When input_size is given, to get front part for padding if True. Default: False.
    """
    def __init__(self, inputs, labels=None, input_size=None, label_size=None, batch_size=128, shuffle=True):
        super(TranslationLoader, self).__init__(inputs, labels=labels, input_size=input_size, batch_size=batch_size, shuffle=shuffle, forward=True)
        self._label_size = label_size

    def _input_treator(self, idx):
        inputs = [self._inputs[i] for i in idx]
        if self._input_size is None:
            size = get_max_size(inputs)
        else:
            size = self._input_size
        return self._padding(inputs, size)

    def _label_treator(self, idx):
        labels = [self._labels[i] for i in idx]
        if self._label_size is None:
            size = get_max_size(labels)
        else:
            size = self._label_size
        return self._padding(labels, size)


class TorchTranslationLoader(TranslationLoader):
    def _padding(self, inputs, size):
        inputs = pad_general(inputs, size, self._forward)
        return torch.LongTensor(inputs)
