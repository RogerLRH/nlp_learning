# -*- coding: utf-8 -*-
import tensorflow as tf

from nlp_learning.tensorflow.function import conv1_layer
from nlp_learning.tensorflow.text_classification.base_model import BaseModel


class TextCNN(BaseModel):
    def __init__(self, dict_size, input_len, num_class, embed_size=100, filter_sizes=[1,2,3,4,5], num_filter=256, l2_ld=1e-4, pos_weight=1., multi_label=False, initial_size=.1):
        self._filter_sizes = filter_sizes
        self._num_filter = num_filter
        self._num_filters = self._num_filter * len(filter_sizes)

        super(TextCNN, self).__init__(dict_size, input_len, num_class, embed_size=embed_size, l2_ld=l2_ld, pos_weight=pos_weight, multi_label=multi_label, initial_size=initial_size)

    def core(self):
        # convolutional
        with tf.name_scope("conv"):
            pool_outs = []
            for i, size in enumerate(self._filter_sizes):
                pooled = conv1_layer(self._embedded_sentence, self._input_len, self._num_filter, size, self._initializer, name="conv_%s"%i)
                pool_outs.append(pooled)
            h_pool = tf.concat(pool_outs, 2) # [None, 1, num_filters].
            h_pool_flat = tf.reshape(h_pool, [-1, self._num_filters])

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self._dropout_keep_prob)

        # fc
        with tf.name_scope("full"):
            W_project = tf.get_variable("W_project", shape=[self._num_filters, self._num_class], initializer=self._initializer)
            b_project = tf.get_variable("bias_project", shape=[self._num_class])
            logits = tf.matmul(h_drop, W_project) + b_project
        return logits
