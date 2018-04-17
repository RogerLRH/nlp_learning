# -*- coding: utf-8 -*-
#TextCNN: 1. embeddding, 2.convolutional, 3.max-pooling, 4.softmax.
import tensorflow as tf

from base_model import BaseModel


class TextCNN(BaseModel):
    def __init__(self, voca_size, input_len, num_class, embed_size=100, filter_sizes=[1,2,3,4,5], num_filter=256, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, batch_size=128, l2_ld=0.0001, pos_weight=1, clip_gradient=5.0, multi_label=False, initial_size=0.1):
        self.filter_sizes = filter_sizes
        self.num_filter = num_filter
        self.num_filters = self.num_filter * len(filter_sizes)

        super(TextCNN, self).__init__(voca_size, input_len, num_class, embed_size, learning_rate, decay_step, decay_rate, batch_size, l2_ld, pos_weight, clip_gradient, multi_label, initial_size)

    def init_weights(self):
        """define all weights here"""
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("conv"):
            self.filters = []
            for size in self.filter_sizes:
                fil = tf.get_variable("filter_%s"%size, [size, self.embed_size, 1, self.num_filter], initializer=self.initializer)
                b = tf.get_variable("b_%s"%size, [self.num_filter])
                self.filters.append((fil, b))

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.num_filters, self.num_class], initializer=self.initializer)
            self.b_project = tf.get_variable("b_project", shape=[self.num_class])

    def core(self):
        # emebedding
        self.expanded = tf.expand_dims(self.embedded_sentence, -1) #input requirement of 2d-conv

        # convolutional
        with tf.name_scope("conv"):
            pool_outs = []
            for i, size in enumerate(self.filter_sizes):
                fil, b = self.filters[i]
                conv = tf.nn.conv2d(self.expanded, fil, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.input_len-size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pool_outs.append(pooled)
            self.h_pool = tf.concat(pool_outs, 3) # [None, 1, 1, num_filters].
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters])

        # dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # softmax
        with tf.name_scope("softmax"):
            self.logits = tf.matmul(self.h_drop, self.W_project) + self.b_project
