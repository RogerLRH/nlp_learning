# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

from base_model import BaseModel


class TextRCNN(BaseModel):
    def __init__(self, voca_size, input_len, hidden_size, num_class, embed_size=100, learning_rate=1e-3, no_decay_step=1000, decay_rate=0.8, batch_size=128, pos_weight=1, initializer=tf.random_normal_initializer(stddev=0.1), multi_label=False, clip_gradient=5.0):
        self.voca_size = voca_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        super(TextRCNN, self).__init__(input_len=input_len, num_class=num_class, learning_rate=learning_rate, no_decay_step=no_decay_step, decay_rate=decay_rate, batch_size=batch_size, pos_weight=pos_weight, clip_gradient=clip_gradient, initializer=initializer, multi_label=multi_label)

    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.hidden_size*2+self.embed_size, self.num_class], initializer=self.initializer)
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
        outs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_sentence, dtype=tf.float32) # tuple([None, input_len, hidden_size])

        # concat output
        output_rnn = tf.concat([outs[0], self.embedded_sentence, outs[1]], axis=2) # [None, input_len, hidden_size*2+embed_size]

        # max pool
        self.output_rnn_last = tf.reduce_max(output_rnn, axis=1)

        # FC
        with tf.name_scope("full"):
            logits = tf.matmul(self.output_rnn_last, self.W_project) + self.b_project  # [None, num_class]
        return logits
