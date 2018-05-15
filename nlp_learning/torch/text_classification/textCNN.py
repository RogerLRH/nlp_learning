# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from nlp_learning.torch.function import conv1_layer


class TextCNN(nn.Module):
    def __init__(self, dict_size, input_len, num_class, filter_sizes=[1, 2, 3, 4, 5], num_filter=256, embed_size=100, use_cuda=True):
        super(TextCNN, self).__init__()
        self._filter_sizes = filter_sizes
        self._num_filter = num_filter
        self._embed_size = embed_size
        self._use_cuda = use_cuda

        self._embedding = nn.Embedding(dict_size, embed_size)
        nn.init.xavier_uniform_(self._embedding.weight)
        self._dropout = nn.Dropout(0.5)

        conv_blocks = []
        for filter_size in filter_sizes:
            conv_struct = conv1_layer(input_len, embed_size, num_filter, filter_size)
            conv_blocks.append(conv_struct)

        self._conv_blocks = nn.ModuleList(conv_blocks)

        self._fc = nn.Linear(num_filter*len(filter_sizes), num_class)
        nn.init.normal_(self._fc.weight)

    def forward(self, inputs):
        embed = self._embedding(inputs)
        embed = self._dropout(embed)

        embed = embed.transpose(1, 2)
        conv_output = [conv_block(embed) for conv_block in self._conv_blocks]

        output = torch.cat(conv_output, 2)
        output = output.view(output.size(0), -1)

        logits = self._fc(output)
        return logits
