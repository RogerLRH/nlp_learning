# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np


class BaseModel(object):
    def __init__(self, voca_size, input_len, num_class, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, batch_size=128, l2_lambda=0.0001, pos_weight=1, clip_gradient=5.0, initializer=tf.random_normal_initializer(stddev=0.1), multi_label=False):
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
        self.l2_lambda = l2_lambda
        self.pos_weight = pos_weight
        self.clip_gradient = clip_gradient
        self.multi_label = multi_label
        self.initializer = initializer

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.build_graph()
        self.saver = tf.train.Saver()

    def build_graph(self):
        self.build_inputs()

        self.init_weights()

        self.embed()

        self.logits = self.core()
        if self.multi_label:
            print("going to use multi label loss.")
            self.loss = self.loss_multi()
        else:
            print("going to use single label loss.")
            self.loss = self.loss_single()
        self.opt = self.optimize()
        self.prob = self.get_prob()

    def build_inputs(self):
        self.input = tf.placeholder(tf.int32, [None, self.input_len], name="input")
        if self.multi_label:
            self.label = tf.placeholder(tf.float32, [None, self.num_class], name="label")
        else:
            self.label = tf.placeholder(tf.float32, [None, 1], name="label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def embed(self):
        self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input) # [None, input_len, embed_size]

    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

    def core(self):
        # main process, return logits
        pass

    def loss_single(self):
        with tf.name_scope("loss"):
            if self.num_class > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits)
            else:
                losses = tf.nn.weighted_cross_entropy_with_logits(
                targets=self.label, logits=self.logits, pos_weight=self.pos_weight)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'b_' not in v.name]) * self.l2_lambda
            loss += l2_losses
        return loss

    def loss_multi(self):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.label, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'b_' not in v.name]) * self.l2_lambda
            loss += l2_losses
        return loss

    def get_prob(self):
        if self.multi_label or self.num_class == 1:
            prob = tf.nn.sigmoid(self.logits)
        else:
            prob = tf.nn.softmax(self.logits)
        return prob

    def calcul_accuracy(self, prob, label):
        if self.multi_label or self.num_class == 1:
            prob[prob >= 0.5] = 1
            prob[prob < 0.5] = 0
        else:
            prob = np.argmax(prob)
        acu = np.mean(prob == label)
        return acu

    def optimize(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_rate, staircase=True)
        op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradient)
        return op

    def train(self, inputs, labels, test_x=None, test_y=None, epochs=5, checkpoint=None, save_path="checkpoints"):
        with tf.Session() as sess:
            if checkpoint:
                self.saver.restore(sess, checkpoint)
            else:
                sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                sf = np.array(range(len(inputs)))
                np.random.shuffle(sf)
                inputs = inputs[sf]
                labels = labels[sf]
                for j in range((len(inputs) - 1) // self.batch_size + 1):
                    train_x = inputs[j*self.batch_size:(j+1)*self.batch_size]
                    train_y = labels[j*self.batch_size:(j+1)*self.batch_size]
                    feed_dict = {self.input: train_x, self.label: train_y, self.dropout_keep_prob: 0.5}
                    sess.run(self.opt,feed_dict)
                    if j % 50 != 0:
                        continue
                    feed_dict[self.dropout_keep_prob] = 1
                    loss, prob = sess.run([self.loss, self.prob], feed_dict=feed_dict)
                    acu = self.calcul_accuracy(prob, train_y)
                    print("Epoch %s, Batch %s, Train:Loss %.6f, Accuracy %.6f" % (i, j, loss, acu))
                    fname = "cp_%s_%s" % (i, j)
                    self.saver.save(sess, os.path.join(save_path, fname))
                    if test_x is None:
                        continue
                    losses, acc = 0, 0
                    for k in range((len(test_x) - 1) // self.batch_size + 1):
                        x = test_x[k*self.batch_size:(k+1)*self.batch_size]
                        y = test_y[k*self.batch_size:(k+1)*self.batch_size]
                        feed_dict[self.input] = x
                        feed_dict[self.label] = y
                        loss, prob = sess.run([self.loss, self.prob], feed_dict=feed_dict)
                        losses += loss
                        acc += self.calcul_accuracy(prob, y)
                    losses = losses * self.batch_size / len(test_x)
                    acc = acc * self.batch_size / len(test_x)
                    print("Test:Loss %.6f, Accuracy %.6f" % (losses, acc))

    def test(self, inputs, labels, checkpoint):
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            losses = 0
            probs = []
            for j in range((len(inputs) - 1) // self.batch_size + 1):
                test_x = inputs[j*self.batch_size:(j+1)*self.batch_size]
                test_y = labels[j*self.batch_size:(j+1)*self.batch_size]
                feed_dict = {self.input: test_x, self.label: test_y, self.dropout_keep_prob: 1}
                loss, prob = sess.run([self.loss, self.prob], feed_dict=feed_dict)
                losses += loss
                probs.append(prob)
            return losses * self.batch_size / len(inputs), np.concatenate(probs)

    def predict(self, inputs, checkpoint):
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            probs = []
            for j in range((len(inputs) - 1) // self.batch_size + 1):
                test_x = inputs[j*self.batch_size:(j+1)*self.batch_size]
                feed_dict = {self.input: test_x, self.dropout_keep_prob: 1}
                prob = sess.run(self.prob, feed_dict=feed_dict)
                probs.append(prob)
            return np.concatenate(probs)
