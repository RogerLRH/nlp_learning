# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding, 2.Bi-LSTM, 3.concat output, 4.FC, 5.softmax
import tensorflow as tf
from tensorflow.contrib import rnn

from base_model import BaseModel


class TextRNN(BaseModel):
    def __init__(self, voca_size, input_len, num_class, hidden_size=100, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, batch_size=128, l2_ld=0.0001, pos_weight=1, clip_gradient=5.0, multi_label=False, initial_size=0.1):
        self.hidden_size = hidden_size

        super(TextRNN, self).__init__(voca_size, input_len, num_class, embed_size, learning_rate, decay_step, decay_rate, batch_size, l2_ld, pos_weight, clip_gradient, multi_label, initial_size)

    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.hidden_size*2, self.num_class], initializer=self.initializer)
            self.bias_project = tf.get_variable("bias_project", shape=[self.num_class])

    def core(self):
        # Bi-lstm
        lstm_fw_cell = rnn.GRUCell(self.hidden_size) #forward direction
        lstm_bw_cell = rnn.GRUCell(self.hidden_size) #backward direction
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_sentence, dtype=tf.float32) # tuple([None, input_len, hidden_size])
        output_rnn = tf.concat(outputs, axis=2) # [None, input_len, hidden_size*2]

        # concat output
        self.h_drop = tf.reduce_mean(output_rnn, axis=1) # [None, hidden_size*2]

        # FC
        with tf.name_scope("full"):
            self.logits = tf.matmul(self.h_drop, self.W_project) + self.bias_project  # [None, num_class]


class TextRNNAttention(TextRNN):
    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("attention"):
            self.W_attention = tf.get_variable("W_attention", shape=[self.hidden_size*2, self.hidden_size*2], initializer=self.initializer)
            self.bias_atteniton = tf.get_variable("bias_atteniton", shape=[self.hidden_size*2])
            self.U_attention = tf.get_variable("attention", shape=[self.hidden_size*2, 1], initializer=self.initializer)

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.hidden_size*2, self.num_class], initializer=self.initializer)
            self.bias_project = tf.get_variable("bias_project", shape=[self.num_class])

    def core(self):
        # Bi-lstm
        lstm_fw_cell = rnn.GRUCell(self.hidden_size) #forward direction
        lstm_bw_cell = rnn.GRUCell(self.hidden_size) #backward direction
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_sentence, dtype=tf.float32) # tuple(2*[None, input_len, hidden_size])
        self.h_rnn = tf.concat(outputs, axis=2) # [None, input_len, hidden_size*2]

        # attention
        input_attention = tf.reshape(self.h_rnn, [-1, self.hidden_size*2])
        hidden_rep = tf.nn.tanh(tf.matmul(input_attention, self.W_attention) + self.bias_atteniton)
        attention_logits = tf.reshape(tf.matmul(hidden_rep, self.U_attention), [-1, self.input_len])
        attention_weight = tf.expand_dims(tf.nn.softmax(attention_logits), axis=2)
        self.h_drop = tf.reduce_sum(tf.multiply(self.h_rnn, attention_weight), axis=1)

        # FC
        with tf.name_scope("full"):
            self.logits = tf.matmul(self.h_drop, self.W_project) + self.bias_project  # [None, num_class]


class TextRNNAttentionWithSentence(TextRNN):
    def build_inputs(self):
        self.input = tf.placeholder(tf.int32, [None, self.input_len[0], self.input_len[1]], name="input")
        if self.multi_label:
            self.label = tf.placeholder(tf.float32, [None, self.num_class], name="label")
        else:
            self.label = tf.placeholder(tf.int32, [None], name="label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("word_attention"):
            self.W_word_attention = tf.get_variable("W_word_attention", shape=[self.hidden_size*2, self.hidden_size*2], initializer=self.initializer)
            self.bias_word_atteniton = tf.get_variable("bias_word_atteniton", shape=[self.hidden_size*2])
            self.U_word_attention = tf.get_variable("U_word_attention", shape=[self.hidden_size*2, 1], initializer=self.initializer)

        with tf.name_scope("sen_attention"):
            self.W_sen_attention = tf.get_variable("W_sen_attention", shape=[self.hidden_size*2, self.hidden_size*2], initializer=self.initializer)
            self.bias_sen_atteniton = tf.get_variable("bias_sen_atteniton", shape=[self.hidden_size*2])
            self.U_sen_attention = tf.get_variable("U_sen_attention", shape=[self.hidden_size*2, 1], initializer=self.initializer)

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.hidden_size*2, self.num_class], initializer=self.initializer)
            self.bias_project = tf.get_variable("bias_project", shape=[self.num_class])

    def core(self):
        # word Bi-lstm
        with tf.variable_scope("word_bi_lstm", reuse=False):
            lstm_fw_cell = rnn.GRUCell(self.hidden_size) #forward direction
            lstm_bw_cell = rnn.GRUCell(self.hidden_size) #backward direction
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            self.embedded_text = tf.reshape(self.embedded_sentence, [-1, self.input_len[1], self.embed_size])
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_text, dtype=tf.float32) # tuple(2*[None, max_time, hidden_size])
            self.word_rnn = tf.concat(outputs, axis=2) # [None, max_time, hidden_size*2]

        # word attention
        input_word_attention = tf.reshape(self.word_rnn, [-1, self.hidden_size*2])
        word_hidden_rep = tf.nn.tanh(tf.matmul(input_word_attention, self.W_word_attention) + self.bias_word_atteniton)
        word_attention_logits = tf.reshape(tf.matmul(word_hidden_rep, self.U_word_attention), [-1, self.input_len[1]])
        word_attention_weight = tf.expand_dims(tf.nn.softmax(word_attention_logits), axis=2)
        self.word_output = tf.reduce_sum(tf.multiply(self.word_rnn, word_attention_weight), axis=1)
        self.sen_input = tf.reshape(self.word_output, [-1, self.input_len[0], self.hidden_size*2])

        # sentence Bi-lstm
        with tf.variable_scope("sen_bi_lstm", reuse=False):
            lstm_fw_cell = rnn.GRUCell(self.hidden_size) #forward direction
            lstm_bw_cell = rnn.GRUCell(self.hidden_size) #backward direction
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.sen_input, dtype=tf.float32) # tuple(2*[None, num_sentence, hidden_size])
            self.sen_rnn = tf.concat(outputs, axis=2) # [None, num_sentence, hidden_size*2]

        # sentence attention
        input_sen_attention = tf.reshape(self.sen_rnn, [-1, self.hidden_size*2])
        sen_hidden_rep = tf.nn.tanh(tf.matmul(input_sen_attention, self.W_sen_attention) + self.bias_sen_atteniton)
        sen_attention_logits = tf.reshape(tf.matmul(sen_hidden_rep, self.U_sen_attention), [-1, self.input_len[0]])
        sen_attention_weight = tf.expand_dims(tf.nn.softmax(sen_attention_logits), axis=2)
        self.sen_output = tf.reduce_sum(tf.multiply(self.sen_rnn, sen_attention_weight), axis=1)

        self.sen_dropout = tf.nn.dropout(self.sen_output, keep_prob=self.dropout_keep_prob)
        # FC
        with tf.name_scope("full"):
            self.logits = tf.matmul(self.sen_dropout, self.W_project) + self.bias_project  # [None, num_class]
