# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding, 2.Bi-LSTM, 3.concat output, 4.FC, 5.softmax
import tensorflow as tf

from nlp_learning.tensorflow.text_classification.base_model import BaseModel
from nlp_learning.tensorflow.function import bi_gru
from nlp_learning.tensorflow.function import attention_layer


class TextRNN(BaseModel):
    def __init__(self, dict_size, input_len, num_class, hidden_size=100, embed_size=100, l2_ld=0.0001, pos_weight=1, multi_label=False, initial_size=0.1):
        self._hidden_size = hidden_size

        super(TextRNN, self).__init__(dict_size, input_len, num_class, embed_size=embed_size, l2_ld=l2_ld, pos_weight=pos_weight, multi_label=multi_label, initial_size=initial_size)

    def core(self):
        # Bi-lstm
        with tf.name_scope("rnn"):
            outputs = bi_gru(self._embedded_sentence, self._hidden_size, self._dropout_keep_prob)
            output_rnn = tf.concat(outputs, axis=2) # [None, input_len, hidden_size*2]

        # concat output
        h_drop = tf.reduce_mean(output_rnn, axis=1) # [None, hidden_size*2]

        # FC
        with tf.name_scope("full"):
            W_project = tf.get_variable("W_project", shape=[self._hidden_size*2, self._num_class], initializer=self._initializer)
            bias_project = tf.get_variable("bias_project", shape=[self._num_class])
            logits = tf.matmul(h_drop, W_project) + bias_project
        return logits


class TextRNNAttention(TextRNN):
    def core(self):
        # Bi-lstm
        with tf.name_scope("rnn"):
            outputs = bi_gru(self._embedded_sentence, self._hidden_size, self._dropout_keep_prob)
            h_rnn = tf.concat(outputs, axis=2) # [None, input_len, hidden_size*2]

        # attention
        with tf.name_scope("attention"):
            h_drop = attention_layer(h_rnn, self._input_len, self._hidden_size, self._initializer)

        # FC
        with tf.name_scope("full"):
            W_project = tf.get_variable("W_project", shape=[self._hidden_size*2, self._num_class], initializer=self._initializer)
            bias_project = tf.get_variable("bias_project", shape=[self._num_class])
            logits = tf.matmul(h_drop, W_project) + bias_project
        return logits


class TextRNNAttentionWithSentence(TextRNN):
    def _build_placeholder(self):
        self._input = tf.placeholder(tf.int32, [None, self._input_len[0], self._input_len[1]], name="input")
        if self._multi_label:
            self._label = tf.placeholder(tf.float32, [None, self._num_class], name="label")
        else:
            self._label = tf.placeholder(tf.int32, [None], name="label")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def core(self):
        # word Bi-lstm
        with tf.variable_scope("word_bi_lstm", reuse=False):
            embedded_text = tf.reshape(self._embedded_sentence, [-1, self._input_len[1], self._embed_size])
            outputs = bi_gru(embedded_text, self._hidden_size, self._dropout_keep_prob)
            word_rnn = tf.concat(outputs, axis=2) # [None, max_time, hidden_size*2]

        # word attention
        with tf.variable_scope("word_attention", reuse=False):
            word_output = attention_layer(word_rnn, self._input_len[1], self._hidden_size, self._initializer)
            sen_input = tf.reshape(word_output, [-1, self._input_len[0], self._hidden_size*2])

        # sentence Bi-lstm
        with tf.variable_scope("sen_bi_lstm", reuse=False):
            outputs = bi_gru(sen_input, self._hidden_size, self._dropout_keep_prob)
            sen_rnn = tf.concat(outputs, axis=2) # [None, num_sentence, hidden_size*2]

        # sentence attention
        with tf.variable_scope("sen_attention", reuse=False):
            sen_output = attention_layer(sen_rnn, self._input_len[0], self._hidden_size, self._initializer)

        # FC
        with tf.name_scope("full"):
            W_project = tf.get_variable("W_project", shape=[self._hidden_size*2, self._num_class], initializer=self._initializer)
            bias_project = tf.get_variable("bias_project", shape=[self._num_class])
            logits = tf.matmul(sen_output, W_project) + bias_project
        return logits
