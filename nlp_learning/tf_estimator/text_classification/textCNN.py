# -*- coding: utf-8 -*-
#TextCNN: 1. embeddding, 2.convolutional, 3.max-pooling, 4.softmax.
import tensorflow as tf

from nlp_learning.tf_estimator.text_classification.base_model import base_model_fn
from nlp_learning.tensorflow.function import conv1_layer


class TextCNN(tf.estimator.Estimator):
    def __init__(
        self,
        dict_size,
        input_len,
        num_class,
        embed_size,
        filter_sizes,
        num_filter,
        learning_rate=1e-3,
        decay_step=1000,
        decay_rate=0.8,
        l2_ld=1e-4,
        pos_weight=1.0,
        clip_gradient=5.,
        multi_label=False,
        initial_size=0.1,
        model_dir=None,
        config=None,
        params=None,
        warm_start_from=None):
        core = CoreTextCNN(input_len, num_class, filter_sizes, num_filter, initial_size=initial_size)

        def _model_fn(features, labels, mode):
            return base_model_fn(
                features,
                labels,
                mode,
                core,
                dict_size,
                num_class,
                embed_size,
                learning_rate,
                decay_step,
                decay_rate,
                l2_ld,
                pos_weight,
                clip_gradient,
                multi_label)

        super(TextCNN, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)


class CoreTextCNN(object):
    def __init__(self, input_len, num_class, filter_sizes, num_filter, initial_size=0.1):
        self.input_len = input_len
        self.num_class = num_class
        self.filter_sizes = filter_sizes
        self.num_filter = num_filter
        self.initializer = tf.random_normal_initializer(stddev=initial_size)

    def __call__(self, embedded_sentence, mode):
        pool_outs = []
        for i, size in enumerate(self.filter_sizes):
            pooled = conv1_layer(embedded_sentence, self.input_len, self.num_filter, size, self.initializer, name="conv_%s"%i)
            pool_outs.append(pooled)
        h_pool = tf.concat(pool_outs, 2) # [None, 1, num_filters].
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filter * len(self.filter_sizes)])

        if mode == tf.estimator.ModeKeys.TRAIN:
            h_pool_flat = tf.layers.dropout(h_pool_flat, name="dropout")

        logits = tf.layers.dense(h_pool_flat, self.num_class, name='dense', kernel_initializer=self.initializer)
        return logits
