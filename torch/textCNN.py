# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    def __init__(self, voca_size, input_len, num_class, filter_sizes=[1,2,3,4,5], num_filter=256, embed_size=100, use_cuda=True):
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filter = num_filter
        self.embed_size = embed_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(voca_size, embed_size)
        nn.init.xavier_uniform(self.embedding.weight)
        self.dropout = nn.Dropout(0.5)

        conv_blocks = []
        for filter_size in filter_sizes:
            maxpool_kernel_size = input_len - filter_size + 1
            conv1 = nn.Conv1d(embed_size, num_filter, filter_size)
            nn.init.xavier_normal(conv1.weight)

            component = nn.Sequential(
                conv1,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size))

            if use_cuda:
                component = component.cuda()
            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.fc = nn.Linear(num_filter*len(filter_sizes), num_class)
        nn.init.normal(self.fc.weight)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        embed = embed.transpose(1, 2)
        conv_output = [conv_block(embed) for conv_block in self.conv_blocks]

        output = torch.cat(conv_output, 2)
        output = output.view(output.size(0), -1)

        logits = F.softmax(self.fc(output), dim=1)
        return logits
