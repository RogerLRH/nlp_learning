# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from nlp_learning.torch.function import get_predict, get_probability, calcul_accuracy
from nlp_learning.torch.function import build_data


class Treator(object):
    def __init__(self, model, multi_label=False, use_cuda=True, parallel=False):
        if use_cuda and parallel and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        elif use_cuda and torch.cuda.is_available():
            model.cuda()

        self._model = model
        self._multi_label = multi_label
        self._use_cuda = use_cuda

        if multi_label:
            self._loss = nn.MultiLabelSoftMarginLoss()
        else:
            self._loss = nn.CrossEntropyLoss()

    def train(self, train_file, save_path, valid_file=None, train_cp=None, batch_size=128, learning_rate=1e-3, epochs=5, l2_ld=1e-4, input_size=500, forward=False):
        train_data = build_data(train_file, input_size, batch_size, forward)
        if valid_file:
            valid_data = build_data(valid_file, input_size, batch_size, forward)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=0)
        self._model.zero_grad()

        if train_cp:
            self._model.load_state_dict(torch.load(train_cp))

        for i in range(epochs):
            for j, data in enumerate(train_data):
                loss, acu = self._batch_process(data, l2_ld, optimizer, "TRAIN")
                if j % 100 != 0:
                    continue
                print("Epoch %s, Batch %s :" %(i, j))
                print("Train: Loss %.6f, Accuracy %.6f" %(loss, acu))
                cp_path = os.path.join(save_path, '%s_%s.cp' %(i, j))
                torch.save(self._model.state_dict(), cp_path)

                if not valid_file:
                    continue
                loss, acu = self._batchs_avg(valid_data, l2_ld)
                print("Validate: Loss %.6f, Accuracy %.6f" % (loss, acu))

    def predict(self, predict_file, batch_size=128, input_size=500, forward=False):
        predict_data = build_data(predict_file, input_size, batch_size, forward, False)
        probs, preds = [], []
        for inputs in predict_data:
            if torch.cuda.is_available() and self._use_cuda:
                inputs = inputs.cuda()
            logits = self._model(inputs)
            predicted = get_predict(logits, multi_label=self._multi_label)
            prob = get_probability(logits, multi_label=self._multi_label)
            probs.append(prob)
            preds.append(predicted)
        return torch.cat(probs), torch.cat(preds)

    def _batch_process(self, data, l2_ld, optimizer=None, mode="TRAIN"):
        inputs, labels = Variable(data[0]), Variable(data[1])
        if torch.cuda.is_available() and self._use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        logits = self._model(inputs)
        predicted = get_predict(logits, multi_label=self._multi_label)

        if self._multi_label:
            loss = self._loss(logits, labels.float())
        else:
            loss = self._loss(logits, labels)
        loss = self._add_l2_reg(loss, l2_ld)
        acu = calcul_accuracy(predicted, labels)

        if mode == "TRAIN":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), .5)
            optimizer.step()
        elif mode not in ("VALIDATE", "TEST"):
            raise ValueError("This mode %s is not defined. Please use one of ('TRAIN', 'VALIDATE', 'TEST')" % mode)
        return loss, acu

    def _batchs_avg(self, dataset, l2_ld):
        losses, acus, total = 0, 0, 0
        for data in dataset:
            length = len(data[0])
            loss, acu = self._batch_process(data, l2_ld, None, "VALIDATE")
            losses += loss * length
            acus += acu * length
            total += length
        losses /= total
        acus /= total
        return losses, acus

    def _add_l2_reg(self, loss, l2_ld):
        l2_ld = torch.tensor(l2_ld)
        l2_reg = torch.tensor(0.)
        for param in list(self._model.parameters())[1:]:
            l2_reg += torch.norm(param)
        loss += l2_ld * l2_reg
        return loss
