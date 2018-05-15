import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from nlp_learning.data_loader import TextDataLoader


def calcul_accuracy(predict, label):
    return np.mean(predict == label)


def optimizer(loss, learning_rate, global_step, decay_rate, decay_step, clip_gradient):
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate, staircase=True)
    opt = tf.contrib.layers.optimize_loss(loss, global_step=global_step, learning_rate=lr, optimizer="Adam", clip_gradients=clip_gradient)
    return opt


def build_dataset(filepath, input_len, batch_size, forward, with_label=True):
    labels = None
    if with_label:
        inputs, labels, _ = pickle.load(open(filepath, "rb"))
    else:
        inputs, _ = pickle.load(open(filepath, "rb"))
    return TextDataLoader(inputs, labels=labels, input_size=input_len, batch_size=batch_size, forward=forward)
