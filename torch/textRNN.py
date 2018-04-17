# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionLayer(nn.Module):
    def __init__(self, input_size, attn_size, use_cuda=True):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(input_size, attn_size)
        self.context = Variable(torch.Tensor(attn_size), requires_grad=True)
        if use_cuda:
            self.context = self.context.cuda()

        nn.init.xavier_normal(self.attn.weight)
        nn.init.normal(self.context)

    def forward(self, inputs): # inputs: [-1, len_seq, input_size]
        rep = torch.tanh(self.attn(inputs))
        logits = torch.matmul(rep, self.context)
        weights = F.softmax(logits, dim=1).view([-1, inputs.shape[1], 1])
        outputs = (weights * inputs).sum(dim=1)
        return outputs


def init_bilstm_hidden(batch_size, hidden_size, use_cuda=True):
    hidden = Variable(torch.zeros(2, batch_size, hidden_size))
    if use_cuda:
        hidden = hidden.cuda()
    return hidden


class TextRNN(nn.Module):
    def __init__(self, voca_size, num_class, hidden_size=100, embed_size=100, use_cuda=True):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform(self.embedding.weight)
        nn.init.normal(self.fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        hidden = init_bilstm_hidden(input_len[0], self.hidden_size, self.use_cuda)
        rnn_output, _ = self.gru(embed, hidden)
        rnn_output = rnn_output.mean(dim=1)

        logits = F.softmax(self.fc(rnn_output), dim=1)
        return logits


# TODO
class TextRCNN(nn.Module):
    def __init__(self, voca_size, num_class, hidden_size=100, embed_size=100, num_filter=256, use_cuda=True):
        super(TextRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform(self.embedding.weight)
        nn.init.normal(self.fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        hidden = init_bilstm_hidden(input_len[0], self.hidden_size, self.use_cuda)
        rnn_output, _ = self.gru(embed, hidden)
        rnn_output = rnn_output.mean(dim=1)

        logits = F.softmax(self.fc(rnn_output), dim=1)
        return logits


class TextRNNAttention(nn.Module):
    def __init__(self, voca_size, num_class, hidden_size=100, embed_size=100, attn_size=100, use_cuda=True):
        super(TextRNNAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.attn = AttentionLayer(hidden_size*2, attn_size, use_cuda)
        self.fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform(self.embedding.weight)
        nn.init.normal(self.fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        hidden = init_bilstm_hidden(input_len[0], self.hidden_size, self.use_cuda)
        rnn_output, _ = self.gru(embed, hidden)
        attn_output = self.attn(rnn_output)

        logits = F.softmax(self.fc(attn_output), dim=1)
        return logits


class TextRNNAttentionWithSentence(nn.Module):
    def __init__(self, voca_size, num_class, hidden_size=100, embed_size=100, attn_size=100, use_cuda=True):
        super(TextRNNAttentionWithSentence, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.word_gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.sen_gru = nn.GRU(hidden_size*2, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self.word_attn = AttentionLayer(hidden_size*2, attn_size, use_cuda)
        self.sen_attn = AttentionLayer(hidden_size*2, attn_size, use_cuda)
        self.fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform(self.embedding.weight)
        nn.init.normal(self.fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self.embedding(inputs.view([-1, input_len[2]]))
        embed = self.dropout(embed)

        hidden = init_bilstm_hidden(input_len[0] * input_len[1], self.hidden_size, self.use_cuda)
        word_rnn_output, _ = self.word_gru(embed, hidden)
        word_attn_output = self.word_attn(word_rnn_output)

        sen_input = word_attn_output.view([-1, input_len[1], self.hidden_size*2])
        sen_input = self.dropout(sen_input)

        hidden = init_bilstm_hidden(input_len[0], self.hidden_size, self.use_cuda)
        sen_rnn_output, _ = self.sen_gru(sen_input, hidden)
        sen_attn_output = self.sen_attn(sen_rnn_output)

        logits = F.softmax(self.fc(sen_attn_output), dim=1)
        return logits
