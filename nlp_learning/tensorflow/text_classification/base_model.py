# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np

from nlp_learning.tensorflow.function import calcul_loss
from nlp_learning.tensorflow.function import calcul_prob
from nlp_learning.tensorflow.function import get_predict
from nlp_learning.tensorflow.function import calcul_accuracy
from nlp_learning.tensorflow.function import build_dataset
from nlp_learning.tensorflow.function import optimizer


class BaseModel(object):
    """
    Base model for text classification.

    Parameters
    ----------
    dict_size : int
        dictionary size
    input_len : int
        Length of each input, can be None in some case, meaning that it depends on batch
    num_class : int
        class number of texts
    embed_size : int
        Default: 100.
    l2_ld : float
        l2 regularization coefficient. Default: 1e-4
    pos_weight : float
        Weight of positive samples, only valid when num_class is 2, to deal with unbalance samples. Default: 1.0
    multi_label : bool
        If a text can have several classes. Default: False
    initial_size : float
        The value for parameters initialization, using tf.random_normal_initializer. Default: 0.1
    """
    def __init__(self, dict_size, input_len, num_class, embed_size=100, l2_ld=1e-4, pos_weight=1., multi_label=False, initial_size=.1):
        tf.reset_default_graph()

        self._dict_size = dict_size
        self._input_len = input_len
        self._num_class = num_class
        self._embed_size = embed_size
        self._multi_label = multi_label

        self._initializer = tf.random_normal_initializer(stddev=initial_size)

        self._build_placeholder()
        self._embed()
        self._logits = self.core()

        self._loss = calcul_loss(multi_label, num_class, self._logits, self._label, l2_ld=l2_ld, pos_weight=pos_weight)

        self._prob = calcul_prob(self._logits, multi_label)

        self._saver = tf.train.Saver()
        self._saver2 = tf.train.Saver([self._embedding])

    def _build_placeholder(self):
        self._input = tf.placeholder(tf.int32, [None, self._input_len], name="input")
        if self._multi_label:
            self._label = tf.placeholder(tf.float32, [None, self._num_class], name="label")
        else:
            self._label = tf.placeholder(tf.int32, [None], name="label")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _embed(self):
        self._embedding = tf.get_variable("embedding", shape=[self._dict_size, self._embed_size], initializer=tf.random_uniform_initializer(maxval=np.sqrt(3)))
        self._embedded_sentence = tf.nn.embedding_lookup(self._embedding, self._input) # [None, input_len, embed_size]

    def core(self):
        # main process, should return logits
        return self._label

    def _batch_process(self, sess, data, opt=None, mode="TRAIN"):
        if mode not in ("TRAIN", "TEST", "VALIDATE"):
            raise ValueError("This mode %s is not defined." % mode)
        feed_dict = {
            self._input: data[0],
            self._label: data[1],
            self._dropout_keep_prob: 1}
        loss, prob = sess.run([self._loss, self._prob], feed_dict=feed_dict)
        predict = get_predict(prob, self._multi_label)

        if mode == "PREDICT":
            return prob, predict

        acu = calcul_accuracy(predict, data[1])

        if mode in ("TEST", "VALIDATE"):
            return loss, acu

        feed_dict[self._dropout_keep_prob] = 0.5
        sess.run(opt, feed_dict)
        return loss, acu

    def _batches_avg(self, sess, dataset):
        losses, acus, total = 0, 0, 0
        for data in dataset:
            loss, acu = self._batch_process(sess, data, mode="VALIDATE")
            length = len(data[0])
            losses += loss * length
            acus += acu * length
            total += length
        losses = losses / total
        acus = acus / total
        return losses, acus

    def train(self, train_file, valid_file=None, epochs=5, batch_size=128, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, clip_gradient=5., checkpoint=None, save_path="checkpoints", forward=False, load_embed_only=False, save_embed_only=False):
        trainset = build_dataset(train_file, self._input_len, batch_size, forward)
        if valid_file:
            validset = build_dataset(valid_file, self._input_len, batch_size, forward)
        opt = optimizer(self._loss, learning_rate, decay_rate, decay_step, clip_gradient)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint:
                if load_embed_only:
                    self._saver2.restore(sess, checkpoint)
                else:
                    self._saver.restore(sess, checkpoint)

            for i in range(epochs):
                for j, data in enumerate(trainset):
                    loss, acu = self._batch_process(sess, data, opt, "TRAIN")

                    if j % 50 != 0:
                        continue
                    print("Epoch %s, Batch %s :" %(i, j))
                    print("Train: Loss %.6f, Accuracy %.6f" % (loss, acu))
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    if save_embed_only:
                        self._saver2.save(sess, os.path.join(save_path, "cp_%s_%s" % (i, j)))
                    else:
                        self._saver.save(sess, os.path.join(save_path, "cp_%s_%s" % (i, j)))

                    if not valid_file:
                        continue
                    loss, acu = self._batches_avg(sess, validset)
                    print("Validate: Loss %.6f, Accuracy %.6f" % (loss, acu))

    def test(self, test_file, checkpoint, batch_size=128, forward=False):
        testset = build_dataset(test_file, self._input_len, batch_size, forward)
        with tf.Session() as sess:
            self._saver.restore(sess, checkpoint)
            loss, acu = self._batches_avg(sess, testset)
            print("Test: Loss %.6f, Accuracy %.6f" % (loss, acu))

    def predict(self, predict_file, checkpoint, batch_size=128, forward=False):
        predictset = build_dataset(predict_file, self._input_len, batch_size, forward, False)
        with tf.Session() as sess:
            self._saver.restore(sess, checkpoint)
            probs = []
            for inputs in predictset:
                feed_dict = {self._input: inputs, self._dropout_keep_prob: 1}
                prob = sess.run(self._prob, feed_dict=feed_dict)
                probs.append(prob)
            return np.concatenate(probs)
