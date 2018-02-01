# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding, 2.Bi-LSTM, 3.concat output, 4.FC, 5.softmax
import tensorflow as tf
from tensorflow.contrib import rnn

from base_model import BaseModel


class TextRNN(BaseModel):
    def __init__(self, voca_size, input_len, hidden_size, num_class, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, batch_size=128, pos_weight=1, clip_gradient=5.0, initializer=tf.random_normal_initializer(stddev=0.1), multi_label=False):
        self.voca_size = voca_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        super(TextRNN, self).__init__(input_len=input_len, num_class=num_class, learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate, batch_size=batch_size, pos_weight=pos_weight, clip_gradient=clip_gradient, initializer=initializer, multi_label=multi_label)

    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.hidden_size*2, self.num_class], initializer=self.initializer)
            self.b_project = tf.get_variable("b_project", shape=[self.num_class])

    def core(self):
        # embedding
        self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input) # [None, input_len, embed_size]

        # Bi-lstm
        lstm_fw_cell = rnn.GRUCell(self.hidden_size) #forward direction
        lstm_bw_cell = rnn.GRUCell(self.hidden_size) #backward direction
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_sentence, dtype=tf.float32) # tuple([None, input_len, hidden_size])
        output_rnn = tf.concat(outputs, axis=2) # [None, input_len, hidden_size*2]

        # concat output
        self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1) # [None, hidden_size*2]

        # FC
        with tf.name_scope("full"):
            logits = tf.matmul(self.output_rnn_last, self.W_project) + self.b_project  # [None, num_class]
        return logits
