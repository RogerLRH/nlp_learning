# -*- coding: utf-8 -*-
import os
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

from nlp_learning.data_loader import TorchTranslationLoader
from nlp_learning.torch.function import get_predict


def calcul_accuracy(predicted, labels):
    size = labels.shape
    corret, total = 0., 0.
    for i in range(size[0]):
        j = 0
        while j < size[1]:
            if labels[i][j] == 0:
                break
            if labels[i][j] == predicted[i][j]:
                corret += 1
            j += 1
            total += 1
    return corret / size[0] / size[1]


def build_data(filepath, input_size=None, label_size=None, batch_size=128, shuffle=True, with_label=True):
    """
    get an endless iterator, which gives batch data, by a file.
    Parameters
    ----------
    filepath : string
        data pickle file.
    input_size : int
        max length of input sentence. Default: None.
    label_size : int
        max length of label sentence. Default: None.
    batch_size : int
        Default: 128.
    with_label : bool
        if data file has labels.

    Returns
    -------
    iterator which gives batch data
    """
    labels = None
    if with_label:
        inputs, labels, _ = pickle.load(open(filepath, "rb"))
    else:
        inputs, _ = pickle.load(open(filepath, "rb"))
    return TorchTranslationLoader(inputs, labels=labels, input_size=input_size, label_size=label_size, batch_size=batch_size, shuffle=shuffle)


class Treator(object):
    def __init__(self, model, use_cuda=False, parallel=False):
        if use_cuda and parallel and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        elif use_cuda and torch.cuda.is_available():
            model.cuda()

        self._model = model
        self._use_cuda = use_cuda

        self._loss = nn.CrossEntropyLoss(ignore_index=0)

    def train(self, train_file, save_path, valid_file=None, train_cp=None, batch_size=128, learning_rate=1e-3, epochs=5, input_size=500, label_size=500):
        train_data = build_data(train_file, input_size, label_size, batch_size)
        if valid_file:
            valid_data = build_data(valid_file, input_size, label_size, batch_size)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=0)
        self._model.zero_grad()

        if train_cp:
            self._model.load_state_dict(torch.load(train_cp))

        for i in range(epochs):
            for j, data in enumerate(train_data):
                loss, acu = self._batch_process(data, optimizer, "TRAIN")
                if j % 100 != 0:
                    continue
                print("Epoch %s, Batch %s :" %(i, j))
                print("Train: Loss %.6f, Accuracy %.6f" %(loss, acu))
                cp_path = os.path.join(save_path, '%s_%s.cp' %(i, j))
                torch.save(self._model.state_dict(), cp_path)

                if not valid_file:
                    continue
                loss, acu = self._batchs_avg(valid_data)
                print("Validate: Loss %.6f, Accuracy %.6f" % (loss, acu))

    def predict(self, predict_file, predict_cp, batch_size=128, input_size=500, label_size=500):
        predict_data = build_data(predict_file, input_size, label_size, batch_size, False, False)
        self._model.load_state_dict(torch.load(predict_cp))
        preds = []
        for inputs in predict_data:
            if torch.cuda.is_available() and self._use_cuda:
                inputs = inputs.cuda()
            logits = self._model(inputs)

            label_size = len(logits)
            predicted = []
            for i in range(label_size):
                predicted.append(get_predict(logits[i]).view(-1, 1))
            predicted = torch.cat(predicted, dim=1)
            preds.append(predicted)
        return torch.cat(preds)

    def _batch_process(self, data, optimizer=None, mode="TRAIN"):
        if mode not in ("TRAIN", "VALIDATE", "TEST"):
            raise ValueError("This mode %s is not defined." % mode)
        inputs, labels = Variable(data[0]), Variable(data[1])
        if torch.cuda.is_available() and self._use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        logits = self._model(inputs, labels)
        label_size = len(logits)
        losses = 0
        predicted = []

        for i in range(label_size):
            losses += self._loss(logits[i], labels[:, i])
            predicted.append(get_predict(logits[i]).view(-1, 1))

        loss = losses / label_size
        predicted = torch.cat(predicted, dim=1)
        acu = calcul_accuracy(predicted, labels)

        if mode == "TRAIN":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), .5)
            optimizer.step()
        return loss, acu

    def _batchs_avg(self, dataset):
        losses, acus, total = 0, 0, 0
        for data in dataset:
            length = len(data[0])
            loss, acu = self._batch_process(data, None, "VALIDATE")
            losses += loss * length
            acus += acu * length
            total += length
        losses /= total
        acus /= total
        return losses, acus
