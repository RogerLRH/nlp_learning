import pickle
import os
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

from base import pad_array, pad_sentence, get_max_size


class TextDataLoader(object):
    def __init__(self, inputs, labels=None, batch_size=128, shuffle=True, forward=False):
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.forward = forward
        self.num = len(self.inputs)

    def __iter__(self):
        return self.loader()

    def loader(self):
        sf = np.array(range(self.num))
        if self.shuffle:
            np.random.shuffle(sf)
        num_batch = (self.num - 1) // self.batch_size + 1
        for i in range(num_batch):
            idx = sf[self.batch_size*i:self.batch_size*(i+1)]
            inputs = [self.inputs[i] for i in idx]
            size = get_max_size(inputs)
            if size[1]:
                inputs = np.array([pad_array(i, size, self.forward) for i in inputs])
            else:
                inputs = np.array([pad_sentence(i, size[0], self.forward) for i in inputs])
            inputs = torch.LongTensor(inputs)
            if self.labels is not None:
                labels = np.array([self.labels[i] for i in idx])
                labels = torch.LongTensor(labels)
                yield inputs, labels
            else:
                yield inputs


class TrainText(Dataset):
    def __init__(self, text_path, size=500, forward=False):
        self.size = size
        self.forward = forward
        self.array = isinstance(self.size, list)
        self.texts, self.labels, _ = pickle.load(open(text_path, "rb"))

    def __getitem__(self, index):
        if self.array:
            text = pad_array(self.texts[index], self.size, self.forward)
        else:
            text = pad_sentence(self.texts[index], self.size, self.forward)
        return text, self.labels[index]

    def __len__(self):
        return len(self.texts)


class TestText(Dataset):
    def __init__(self, text_path, size=500, forward=False):
        self.size = size
        self.forward = forward
        self.array = isinstance(self.size, list)
        self.texts, _ = pickle.load(open(text_path, "rb"))

    def __getitem__(self, index):
        if self.array:
            text = pad_array(self.texts[index], self.size, self.forward)
        else:
            text = pad_sentence(self.texts[index], self.size, self.forward)
        return text

    def __len__(self):
        return len(self.texts)


def calcul_accuracy(logits, labels, multi_label):
    if multi_label:
        predicted = (logits.data.sigmoid() >= 0.5).long()
    else:
        _, predicted = torch.max(logits.data, 1)
    return (predicted == labels.data).sum() / labels.data.numel()


class Treator(object):
    def __init__(self, model, multi_label=False, use_cuda=True):
        # if torch.cuda.device_count() > 1 and use_cuda:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()

        self.model = model
        self.multi_label = multi_label
        self.use_cuda = use_cuda

        if multi_label:
            self.loss = nn.MultiLabelSoftMarginLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def train(self, train_file, save_path, valid_file=None, train_cp=None, batch_size=128, learning_rate=1e-3, epochs=5, l2_lambda=1e-4, data_size=500, forward=False):
        if not data_size:
            inputs, labels, _ = pickle.load(open(train_file, "rb"))
            train_data = TextDataLoader(inputs, labels=labels, batch_size=batch_size, shuffle=True, forward=forward)
            if valid_file:
                inputs, labels, _ = pickle.load(open(valid_file, "rb"))
                valid_data = TextDataLoader(inputs, labels=labels, batch_size=batch_size, shuffle=True, forward=forward)
        else:
            train_data = DataLoader(TrainText(train_file, data_size, forward=forward), batch_size=batch_size, shuffle=True)
            if valid_file:
                valid_data = DataLoader(TrainText(valid_file, data_size, forward=forward), batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
        self.model.zero_grad()

        if train_cp:
            self.model.load_state_dict(torch.load(train_cp))

        for i in range(epochs):
            for j, data in enumerate(train_data):
                inputs, labels = Variable(data[0]), Variable(data[1])
                if torch.cuda.is_available() and self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                logits = self.model(inputs)
                if self.multi_label:
                    loss = self.loss(logits, labels.float())
                else:
                    loss = self.loss(logits, labels)
                loss.backward()
                optimizer.step()

                if j % 100 != 0:
                    continue

                print("Epoch %s, Batch %s :" %(i, j))
                acu = calcul_accuracy(logits, labels, self.multi_label)
                print("Train: Loss %.6f, Accuracy %.6f" %(loss, acu))
                cp_path = os.path.join(save_path, '%s_%s.cp' %(i, j))
                torch.save(self.model.state_dict(), cp_path)

                if not valid_file:
                    continue

                loss, acu, total = 0, 0, 0
                for k, data in enumerate(valid_data):
                    inputs, labels = Variable(data[0]), Variable(data[1])
                    if torch.cuda.is_available() and self.use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    logits = self.model(inputs)
                    loss += self.loss(logits, labels) * len(inputs)
                    acu += calcul_accuracy(logits, labels, self.multi_label) * len(inputs)
                    total += len(inputs)
                loss /= total
                acu /= total
                print("Validate: Loss %.6f, Accuracy %.6f" % (loss, acu))

    def predict(self, predict_file, batch_size=128, data_size=500, forward=False):
        if not data_size:
            inputs, _ = pickle.load(open(predict_file, "rb"))
            predict_data = TextDataLoader(inputs, batch_size=batch_size, shuffle=True, forward=forward)
        else:
            predict_data = DataLoader(TestText(predict_file, data_size, forward=forward), batch_size=batch_size)
        probs, preds = [], []
        for data in predict_data:
            inputs, labels = Variable(data[0]), Variable(data[1])
            if torch.cuda.is_available() and self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            logits = self.model(inputs)
            probs.append(logits.data)
            if self.multi_label:
                predicted = logits.data >= 0.5
            else:
                _, predicted = torch.max(logits.data, 1)
            preds.append(predicted)
        return torch.cat(probs), torch.cat(preds)
