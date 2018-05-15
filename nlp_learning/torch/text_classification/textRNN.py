# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nlp_learning.torch.function import init_const, conv1_layer


class AttentionLayer(nn.Module):
    def __init__(self, input_size, attn_size, use_cuda=True):
        super(AttentionLayer, self).__init__()
        self._attn = nn.Linear(input_size, attn_size)
        self._context = Variable(torch.Tensor(attn_size), requires_grad=True)
        if use_cuda:
            self._context = self._context.cuda()

        nn.init.xavier_normal_(self._attn.weight)
        nn.init.normal_(self._context)

    def forward(self, inputs): # inputs: [-1, len_seq, input_size]
        rep = torch.tanh(self._attn(inputs))
        logits = torch.matmul(rep, self._context)
        weights = F.softmax(logits, dim=1).view([-1, inputs.shape[1], 1])
        outputs = (weights * inputs).sum(dim=1)
        return outputs


class TextRNN(nn.Module):
    def __init__(self, dict_size, num_class, hidden_size=100, embed_size=100, use_cuda=True):
        super(TextRNN, self).__init__()
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._use_cuda = use_cuda

        self._embedding = nn.Embedding(dict_size, embed_size)
        self._dropout = nn.Dropout(0.5)
        self._gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self._fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform_(self._embedding.weight)
        nn.init.normal_(self._fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self._embedding(inputs)
        embed = self._dropout(embed)

        hidden = init_const(2, input_len[0], self._hidden_size, use_cuda=self._use_cuda)
        rnn_output, _ = self._gru(embed, hidden)
        rnn_output = rnn_output.mean(dim=1)

        logits = self._fc(rnn_output)
        return logits


class TextRCNN(nn.Module):
    def __init__(self, dict_size, input_len, num_class, hidden_size=100, embed_size=100, num_filter=256, use_cuda=True):
        super(TextRCNN, self).__init__()
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._use_cuda = use_cuda

        self._embedding = nn.Embedding(dict_size, embed_size)
        self._dropout = nn.Dropout(0.5)
        self._gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self._conv1 = conv1_layer(input_len, hidden_size*2, num_filter, 1)
        self._fc = nn.Linear(num_filter, num_class)

        nn.init.xavier_uniform_(self._embedding.weight)
        nn.init.normal_(self._fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self._embedding(inputs)
        embed = self._dropout(embed)

        hidden = init_const(2, input_len[0], self._hidden_size, use_cuda=self._use_cuda)
        rnn_output, _ = self._gru(embed, hidden)
        rnn_output = rnn_output.transpose(1, 2)
        conv_output = self._conv1(rnn_output).view([input_len[0], -1])

        logits = self._fc(conv_output)
        return logits


class TextRNNAttention(nn.Module):
    def __init__(self, dict_size, num_class, hidden_size=100, embed_size=100, attn_size=100, use_cuda=True):
        super(TextRNNAttention, self).__init__()
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._use_cuda = use_cuda

        self._embedding = nn.Embedding(dict_size, embed_size)
        self._dropout = nn.Dropout(0.5)
        self._gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self._attn = AttentionLayer(hidden_size*2, attn_size, use_cuda)
        self._fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform_(self._embedding.weight)
        nn.init.normal_(self._fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self._embedding(inputs)
        embed = self._dropout(embed)

        hidden = init_const(2, input_len[0], self._hidden_size, use_cuda=self._use_cuda)
        rnn_output, _ = self._gru(embed, hidden)
        attn_output = self._attn(rnn_output)

        logits = self._fc(attn_output)
        return logits


class TextRNNAttentionWithSentence(nn.Module):
    def __init__(self, dict_size, num_class, hidden_size=100, embed_size=100, attn_size=100, use_cuda=True):
        super(TextRNNAttentionWithSentence, self).__init__()
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._use_cuda = use_cuda

        self._embedding = nn.Embedding(dict_size, embed_size)
        self._dropout = nn.Dropout(0.5)
        self._word_gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self._sen_gru = nn.GRU(hidden_size*2, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        self._word_attn = AttentionLayer(hidden_size*2, attn_size, use_cuda)
        self._sen_attn = AttentionLayer(hidden_size*2, attn_size, use_cuda)
        self._fc = nn.Linear(hidden_size*2, num_class)

        nn.init.xavier_uniform_(self._embedding.weight)
        nn.init.normal_(self._fc.weight)

    def forward(self, inputs):
        input_len = inputs.shape
        embed = self._embedding(inputs.view([-1, input_len[2]]))
        embed = self._dropout(embed)

        hidden = init_const(2, input_len[0] * input_len[1], self._hidden_size, use_cuda=self._use_cuda)
        word_rnn_output, _ = self._word_gru(embed, hidden)
        word_attn_output = self._word_attn(word_rnn_output)

        sen_input = word_attn_output.view([-1, input_len[1], self._hidden_size*2])
        sen_input = self._dropout(sen_input)

        hidden = init_const(2, input_len[0], self._hidden_size, use_cuda=self._use_cuda)
        sen_rnn_output, _ = self._sen_gru(sen_input, hidden)
        sen_attn_output = self._sen_attn(sen_rnn_output)

        logits = self._fc(sen_attn_output)
        return logits
