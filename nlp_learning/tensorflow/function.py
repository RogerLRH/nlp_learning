import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from nlp_learning.data_loader import ClassLoader


def calcul_loss(multi, num_class, logits, labels, l2_ld=0.0001, pos_weight=1):
    if multi:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
    else:
        if num_class > 2:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
        else:
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=tf.to_float(labels), logits=logits[:,1]-logits[:,0], pos_weight=pos_weight, name="loss")
    loss = tf.reduce_mean(loss)
    l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and "embedding" not in v.name]) * l2_ld
    loss += l2_losses
    return loss


def calcul_prob(logits, multi_label):
    if multi_label:
        prob = tf.nn.sigmoid(logits)
    else:
        prob = tf.nn.softmax(logits)
    return prob


def get_predict(prob, multi_label):
    if multi_label:
        predict = np.zeros_like(prob)
        predict[prob >= 0.5] = 1
    else:
        predict = np.argmax(prob, axis=1)
    return predict


def calcul_accuracy(predict, label):
    return np.mean(predict == label)


def optimizer(loss, learning_rate, decay_rate, decay_step, clip_gradient):
    if tf.train.get_global_step() is None:
        tf.Variable(0, trainable=False, name="global_step")
    lr = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), decay_step, decay_rate, staircase=True)
    opt = tf.contrib.layers.optimize_loss(loss, global_step=tf.train.get_global_step(), learning_rate=lr, optimizer="Adam", clip_gradients=clip_gradient)
    return opt


def build_dataset(filepath, input_len, batch_size, forward, with_label=True):
    labels = None
    if with_label:
        inputs, labels, _ = pickle.load(open(filepath, "rb"))
    else:
        inputs, _ = pickle.load(open(filepath, "rb"))
    return ClassLoader(inputs, labels=labels, input_size=input_len, batch_size=batch_size, forward=forward)


def conv1_layer(inputs, input_len, num_filter, kernel_size, initializer, name=None):
    conv_output = tf.layers.conv1d(inputs, num_filter, kernel_size, name=name, activation=tf.nn.relu, kernel_initializer=initializer)
    pool_name = None
    if name is not None:
        pool_name = name+"_pool"
    pooled = tf.layers.max_pooling1d(conv_output, input_len-kernel_size+1, 1, name=pool_name)
    return pooled


def bi_gru(inputs, hidden_size, dropout_keep_prob):
    lstm_fw_cell = rnn.GRUCell(hidden_size) #forward direction
    lstm_bw_cell = rnn.GRUCell(hidden_size) #backward direction
    if dropout_keep_prob is not None:
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_keep_prob)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_keep_prob)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)
    return outputs


def attention_layer(inputs, input_len, hidden_size, initializer):
    W_attention = tf.get_variable("W_attention", shape=[hidden_size*2, hidden_size*2], initializer=initializer)
    bias_atteniton = tf.get_variable("bias_atteniton", shape=[hidden_size*2])
    U_attention = tf.get_variable("attention", shape=[hidden_size*2, 1], initializer=initializer)

    input_attention = tf.reshape(inputs, [-1, hidden_size*2])
    hidden_rep = tf.nn.tanh(tf.matmul(input_attention, W_attention) + bias_atteniton)
    attention_logits = tf.reshape(tf.matmul(hidden_rep, U_attention), [-1, input_len])
    attention_weight = tf.expand_dims(tf.nn.softmax(attention_logits), axis=2)
    return tf.reduce_sum(tf.multiply(inputs, attention_weight), axis=1)
