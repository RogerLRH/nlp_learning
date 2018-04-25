# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def train_input_fn(features, labels, batch_size, total_num):
    # features and labels are numpy array
    dataset = tf.data.Dataset.from_tensor_slices(({"x": features}, labels))
    return dataset.shuffle(total_num).repeat().batch(batch_size)


def calcul_prob(logits, multi_label, num_class):
    if multi_label or num_class == 2:
        prob = tf.nn.sigmoid(logits)
    else:
        prob = tf.nn.softmax(logits)
    return prob


def calcul_loss(multi, num_class, logits, labels, l2_ld=0.0001, pos_weight=1):
    if multi:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
    else:
        if num_class > 2:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
        else:
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=pos_weight, name="loss")
    loss = tf.reduce_mean(loss)
    l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and "embedding" not in v.name]) * l2_ld
    loss += l2_losses
    return loss


def calcul_accuracy(prob, labels, multi_label, num_class):
    if multi_label or num_class == 2:
        prob[prob >= 0.5] = 1
        prob[prob < 0.5] = 0
    else:
        prob = np.argmax(prob, axis=1)
    acu = np.mean(prob == labels)
    return acu


def make_optimizer(loss, learning_rate, decay_step, decay_rate):
    global_step = tf.Variable(0)
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate)
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)


def conv_layer(inputs, input_len, num_filter, kernel_size, initializer):
    conv_output = tf.layers.conv1d(inputs, num_filter, kernel_size, name="conv", activation=tf.nn.relu, kernel_initializer=initializer)
    pooled = tf.layers.max_pooling1d(conv_output, input_len-kernel_size+1, 1, name="pool")
    return pooled


def base_model_fn(features, labels, mode, core, voca_size, num_class, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, l2_ld=1e-4, pos_weight=1.0, multi_label=False):
    embedding = tf.get_variable("embedding", shape=[voca_size, embed_size], initializer=tf.random_uniform_initializer(maxval=np.sqrt(3)))
    embedded_sentence = tf.nn.embedding_lookup(embedding, features["x"])

    logits = core(embedded_sentence, mode)

    prob = calcul_prob(logits, multi_label, num_class)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"prob": prob})

    loss = calcul_loss(multi_label, num_class, logits, labels, l2_ld=l2_ld, pos_weight=pos_weight)

    acu = calcul_accuracy(prob, labels, multi_label, num_class)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={"acu": acu})

    optimizer = make_optimizer(loss, learning_rate, decay_step, decay_rate)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=optimizer, eval_metric_ops={"acu": acu})
