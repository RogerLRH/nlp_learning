# -*- coding: utf-8 -*-
import tensorflow as tf

from nlp_learning.tensorflow.text_classification.base_model import BaseModel
from nlp_learning.tensorflow.text_classification.textCNN import conv_layer
from nlp_learning.tensorflow.text_classification.textRNN import bi_gru


class TextRCNN(BaseModel):
    def __init__(self, dict_size, input_len, num_class, hidden_size=100, embed_size=100, num_filter=256, l2_ld=0.0001, pos_weight=1, multi_label=False, initial_size=0.1):
        self._hidden_size = hidden_size
        self._conv_size = hidden_size * 2 + embed_size
        self._num_filter = num_filter

        super(TextRCNN, self).__init__(dict_size, input_len, num_class, embed_size=embed_size, l2_ld=l2_ld, pos_weight=pos_weight, multi_label=multi_label, initial_size=initial_size)

    def core(self):
        # Bi-lstm
        with tf.name_scope("rnn"):
            outs = bi_gru(self._embedded_sentence, self._hidden_size, self._dropout_keep_prob)
            output_rnn = tf.concat([outs[0], self._embedded_sentence, outs[1]], axis=2) # [None, input_len, conv_size]

        # convolutional
        with tf.name_scope("conv"):
            pooled = conv_layer(output_rnn, self._input_len, self._num_filter, 1, self._initializer, name="conv")
            h_drop = tf.reshape(pooled, [-1, self._num_filter])

        # FC
        with tf.name_scope("full"):
            W_project = tf.get_variable("W_project", shape=[self._num_filter, self._num_class], initializer=self._initializer)
            b_project = tf.get_variable("bias_project", shape=[self._num_class])
            self._logits = tf.matmul(h_drop, W_project) + b_project
