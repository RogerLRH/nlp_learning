# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_learning.torch.function import init_const
from nlp_learning.const import TOKEN


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.5, use_cuda=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        embedded = self.embedding(inputs).transpose(0, 1)
        embedded = self.dropout(embedded)
        hidden = init_const(2, batch_size, self.hidden_size, use_cuda=self.use_cuda)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, max_output_len, output_size, embed_size, hidden_size, attn_size, dropout_p=0.5, use_cuda=False):
        super(DecoderRNN, self).__init__()
        self.max_output_len = max_output_len
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = nn.Linear(hidden_size * 3, attn_size)
        self.attn_c = nn.Linear(attn_size, 1, bias=False)
        self.gru = nn.GRU(embed_size+hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_step(self, inputs, hidden, encoder_output):
        batch_size = inputs.shape[0]
        embedded = self.embedding(input).view(1, batch_size, -1) #[1, batch_size, embed_size]
        embedded = self.dropout(embedded)

        encoder_len = encoder_output.shape[0]
        hidden_extend = torch.cat([hidden]*encoder_len) #[encoder_len, batch_size, hidden_size]
        attn_tanh = torch.tanh(self.attn(torch.cat((hidden_extend, encoder_output), 2))) #[encoder_len, batch_size, attn_size]
        attn_weights = F.softmax(self.attn_c(attn_tanh), dim=0) #[encoder_len, batch_size, 1]
        attn_output = torch.mean(encoder_len*attn_weights, dim=0, keepdim=True) #[1, batch_size, hidden_size*2]

        gru_input = torch.cat((embedded, attn_output), 2) #[1, batch_size, hidden_size*2+embed_size]
        gru_output, hidden = self.gru(gru_input, hidden)

        output = F.log_softmax(self.out(gru_output[0]), dim=1) #[batch_size, output_size]
        return output, hidden

    def forward(self, encoder_hidden, encoder_output, targets=None):
        batch_size = encoder_output.shape[1]
        max_len = self.max_output_len
        if targets is not None:
            max_len = targets.shape[1]

        decoder_input = init_const(batch_size, value=TOKEN.START, dtype=torch.long, use_cuda=self.use_cuda)
        hidden = encoder_hidden[1].unsqueeze(0)
        decoder_outputs = []

        use_teacher_forcing = False if (targets is None or random.random() < 0.5) else True

        for i in range(max_len):
            decoder_output, hidden = self.forward_step(decoder_input, hidden, encoder_output)
            decoder_outputs.append(decoder_output)
            if use_teacher_forcing:
                decoder_input = targets[:, i]
            else:
                _, top_index = decoder_output.topk(1)
                decoder_input = top_index
        return torch.cat(decoder_outputs, dim=0)


class Seq2seq(nn.Module):
    def __init__(self, input_size, output_size, max_output_len, embed_size, hidden_size, attn_size, dropout_p=0.5, use_cuda=False):
        super(Seq2seq, self).__init__()
        self.encoder = EncoderRNN(input_size=input_size, embed_size=embed_size, hidden_size=hidden_size, dropout_p=dropout_p, use_cuda=use_cuda)
        self.decoder = DecoderRNN(max_output_len=max_output_len, output_size=output_size, embed_size=embed_size, hidden_size=hidden_size, attn_size=attn_size, dropout_p=dropout_p, use_cuda=use_cuda)

    def forward(self, inputs, targets=None):
        encoder_output, encoder_hidden = self.encoder(inputs)
        decoder_output = self.decoder(encoder_hidden, encoder_output, targets)
        return decoder_output
