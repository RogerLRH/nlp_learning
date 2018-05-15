# -*- coding: utf-8 -*-
import tensorflow as tf

from nlp_learning.tensorflow.text_classification.base_model import BaseModel


class FastText(BaseModel):
    def __init__(self, dict_size, input_len, num_class, embed_size=100, l2_ld=0.0001, pos_weight=1, multi_label=False, initial_size=0.1):
        # set hyperparamter
        self._dict_size = dict_size

        super(FastText, self).__init__(dict_size, input_len, num_class, embed_size=embed_size, l2_ld=l2_ld, pos_weight=pos_weight, multi_label=multi_label, initial_size=initial_size)

    def core(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # emebedding
        embedded_sentence = tf.nn.embedding_lookup(self._embedding, self._input)  # [None, self._input_len, self._embed_size]

        # average of vectors and relu
        embeds = tf.reduce_mean(embedded_sentence, axis=1)
        h = tf.nn.relu(embeds, name="relu")

        # fc
        with tf.name_scope("full"):
            W_project = tf.get_variable("W_project", shape=[self._embed_size, self._num_class], initializer=self._initializer)
            b_project = tf.get_variable("bias_project", shape=[self._num_class])
            self._logits = tf.matmul(h, W_project) + b_project
