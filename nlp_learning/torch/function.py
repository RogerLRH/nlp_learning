# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def init_const(*size, **kwargs):
    value = kwargs.get("value", 0)
    dtype = kwargs.get("dtype", None)
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


# TODO：make it runable
class SampledSoftmaxLossWithLogits(nn.Module):
    def __init__(self, num_sampled):
        super(SampledSoftmaxLossWithLogits, self).__init__()
        self._num_sampled = num_sampled
        self._loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, logits, labels):
        batch = logits.shape[0]
        num_class = logits.shape[1]
        logits = logits.data
        sampled_logits = []
        for i in range(batch):
            sampled = logits[i, np.random.choice(num_class, self._num_sampled, replace=False)]
            sampled = torch.cat((logits[i, labels[i]].view(1), sampled))
            sampled_logits.append(sampled.view(1, -1))
        sampled_logits = torch.cat(sampled_logits, dim=0)
        return self._loss(sampled_logits, torch.zeros_like(labels, dtype=torch.long))
