# -*- coding: utf-8 -*-
import pickle

import torch
import torch.nn as nn

from nlp_learning.data_loader import TorchTextDataLoader


def init_const(*size, **kwargs):
    value = kwargs.get("value", 0)
    dtype = kwargs.get("value", None)
    use_cuda = kwargs.get("use_cuda", True)
    const = value * torch.ones(*size, dtype=dtype)
    if use_cuda:
        const = const.cuda()
    return const


def conv1_layer(input_len, embed_size, num_filter, filter_size):
    maxpool_kernel_size = input_len - filter_size + 1

    conv1 = nn.Conv1d(embed_size, num_filter, filter_size)
    nn.init.xavier_normal_(conv1.weight)

    conv_struct = nn.Sequential(
        conv1,
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=maxpool_kernel_size))
    return conv_struct


def build_data(filepath, input_size, batch_size, forward, with_label=True):
    labels = None
    if with_label:
        inputs, labels, _ = pickle.load(open(filepath, "rb"))
    else:
        inputs, _ = pickle.load(open(filepath, "rb"))
    return TorchTextDataLoader(inputs, labels=labels, input_size=input_size, batch_size=batch_size, shuffle=True, forward=forward)


def get_probability(logits, multi_label=False):
    if multi_label:
        prob = logits.data.sigmoid()
    else:
        prob = logits.data.softmax()
    return prob


def get_predict(logits, multi_label=False):
    if multi_label:
        predicted = (logits.data.sigmoid() >= 0.5).long()
    else:
        _, predicted = torch.max(logits.data, 1)
    return predicted


def calcul_accuracy(predicted, labels):
    return (predicted == labels.data).float().sum() / labels.data.numel()
