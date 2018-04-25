# -*- coding: utf-8 -*-
import os
import pickle
import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np

from base import TextIndexLoader


class BaseModel(object):
    def __init__(self, voca_size, input_len, num_class, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, batch_size=128, l2_ld=0.0001, pos_weight=1, clip_gradient=5.0, multi_label=False, initial_size=0.1):
        tf.reset_default_graph()

        # set hyperparamter
        self.voca_size = voca_size
        self.input_len = input_len
        self.num_class = 1 if num_class == 2 else num_class
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.l2_lambda = l2_ld
        self.pos_weight = pos_weight
        self.clip_gradient = clip_gradient
        self.multi_label = multi_label
        self.initializer = tf.random_normal_initializer(stddev=initial_size)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.build_graph()
        self.saver = tf.train.Saver()
        self.saver2 = tf.train.Saver([self.embedding])

    def build_graph(self):
        self.build_placeholder()
        self.init_weights()
        self.embed()
        self.core()

        if self.multi_label:
            print("going to use multi label loss.")
            self.loss = self.loss_multi()
        else:
            print("going to use single label loss.")
            self.loss = self.loss_single()

        self.optimize()
        self.get_prob()

    def build_placeholder(self):
        self.input = tf.placeholder(tf.int32, [None, self.input_len], name="input")
        if self.multi_label:
            self.label = tf.placeholder(tf.float32, [None, self.num_class], name="label")
        else:
            self.label = tf.placeholder(tf.int32, [None], name="label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def embed(self):
        self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input) # [None, input_len, embed_size]

    def init_weights(self):
        # define all weights here
        self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=tf.random_uniform_initializer(maxval=np.sqrt(3)))

    def core(self):
        # main process, define self.logits
        self.logits = self.label

    def loss_single(self):
        with tf.name_scope("loss"):
            if self.num_class > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits)
            else:
                target = tf.to_float(tf.reshape(self.label, [-1, 1]))
                losses = tf.nn.weighted_cross_entropy_with_logits(
                targets=target, logits=self.logits, pos_weight=self.pos_weight)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and "embedding" not in v.name]) * self.l2_lambda
            loss += l2_losses
        return loss

    def loss_multi(self):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.label, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and "embedding" not in v.name]) * self.l2_lambda
            loss += l2_losses
        return loss

    def get_prob(self):
        if self.multi_label or self.num_class == 1:
            self.prob = tf.nn.sigmoid(self.logits)
        else:
            self.prob = tf.nn.softmax(self.logits)

    def calcul_accuracy(self, prob, label):
        if self.multi_label or self.num_class == 1:
            prob[prob >= 0.5] = 1
            prob[prob < 0.5] = 0
        else:
            prob = np.argmax(prob, axis=1)
        acu = np.mean(prob == label)
        return acu

    def optimize(self):
        lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_rate, staircase=True)
        self.opt = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=lr, optimizer="Adam", clip_gradients=self.clip_gradient)

    def train(self, train_file, valid_file=None, epochs=5, checkpoint=None, save_path="checkpoints", forward=False, embed_only=False):
        inputs, labels, _ = pickle.load(open(train_file, "rb"))
        trainset = TextIndexLoader((inputs, labels), self.batch_size, fix_size=self.input_len, forward=forward)
        if valid_file:
            inputs, labels, _ = pickle.load(open(valid_file, "rb"))
            validset = TextIndexLoader((inputs, labels), self.batch_size, fix_size=self.input_len, forward=forward)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint:
                if embed_only:
                    self.saver2.restore(sess, checkpoint)
                else:
                    self.saver.restore(sess, checkpoint)

            for i in range(epochs):
                for j, (inputs, labels) in enumerate(trainset):
                    feed_dict = {self.input: inputs, self.label: labels, self.dropout_keep_prob: 0.5}
                    sess.run(self.opt, feed_dict)
                    if j % 50 != 0:
                        continue
                    print("Epoch %s, Batch %s :" %(i, j))
                    feed_dict[self.dropout_keep_prob] = 1
                    loss, prob = sess.run([self.loss, self.prob], feed_dict=feed_dict)
                    acu = self.calcul_accuracy(prob, labels)
                    print("Train: Loss %.6f, Accuracy %.6f" % (loss, acu))
                    fname = "cp_%s_%s" % (i, j)
                    self.saver.save(sess, os.path.join(save_path, fname))
                    if not valid_file:
                        continue
                    losses, acc = 0, 0
                    for k, (inputs, labels) in enumerate(validset):
                        feed_dict[self.input] = inputs
                        feed_dict[self.label] = labels
                        loss, prob = sess.run([self.loss, self.prob], feed_dict=feed_dict)
                        losses += loss * len(inputs)
                        acc += self.calcul_accuracy(prob, labels) * len(inputs)
                    losses = losses / len(validset)
                    acc = acc / len(validset)
                    print("Validate: Loss %.6f, Accuracy %.6f" % (losses, acc))

    def test(self, test_file, checkpoint, forward=False):
        inputs, labels, _ = pickle.load(open(test_file, "rb"))
        testset = TextIndexLoader((inputs, labels), self.batch_size, fix_size=self.input_len, forward=forward)
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            losses, acc = 0, 0
            probs = []
            for j, (inputs, labels) in enumerate(testset):
                feed_dict = {self.input: inputs, self.label: labels, self.dropout_keep_prob: 1}
                loss, prob = sess.run([self.loss, self.prob], feed_dict=feed_dict)
                losses += loss * len(inputs)
                acc += self.calcul_accuracy(prob, labels) * len(inputs)
            losses = losses / len(testset)
            acc = acc / len(testset)
            print("Test: Loss %.6f, Accuracy %.6f" % (losses, acc))

    def predict(self, predict_file, checkpoint, forward=False):
        inputs, _ = pickle.load(open(predict_file, "rb"))
        predictset = TextIndexLoader(inputs, self.batch_size, fix_size=self.input_len, forward=forward)
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            probs = []
            for j, inputs in enumerate(predictset):
                feed_dict = {self.input: inputs, self.dropout_keep_prob: 1}
                prob = sess.run(self.prob, feed_dict=feed_dict)
                probs.append(prob)
            return np.concatenate(probs)
