# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from nlp_learning.tensorflow.function import calcul_loss
from nlp_learning.tensorflow.function import calcul_prob
from nlp_learning.tensorflow.function import optimizer


def get_predict(prob, multi_label):
    if multi_label:
        predict = tf.where(tf.greater(prob, 0.5), tf.ones_like(prob), tf.zeros_like(prob))
    else:
        predict = np.argmax(prob, axis=1)
    return predict


def train_input_fn(features, labels, batch_size, total_num):
    # features and labels are numpy array
    dataset = tf.data.Dataset.from_tensor_slices(({"x": features}, labels))
    return dataset.shuffle(total_num).repeat().batch(batch_size)


def base_model_fn(features, labels, mode, core, dict_size, num_class, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, l2_ld=1e-4, pos_weight=1.0, clip_gradient=5., multi_label=False):
    embedding = tf.get_variable("embedding", shape=[dict_size, embed_size], initializer=tf.random_uniform_initializer(maxval=np.sqrt(3)))
    embedded_sentence = tf.nn.embedding_lookup(embedding, features["x"])

    logits = core(embedded_sentence, mode)

    prob = calcul_prob(logits, multi_label)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"prob": prob})

    if multi_label:
        labels = tf.to_float(labels)
    loss = calcul_loss(multi_label, num_class, logits, labels, l2_ld=l2_ld, pos_weight=pos_weight)

    pred = get_predict(prob, multi_label)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={"acu": tf.metrics.accuracy(labels, pred)})

    opt = optimizer(loss, learning_rate, decay_rate, decay_step, clip_gradient)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=opt, eval_metric_ops={"acu": tf.metrics.accuracy(labels, pred)})
