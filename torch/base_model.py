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
    def __init__(self, dataset, batch_size, shuffle=True):
        if len(dataset) == 2:
            self.inputs = dataset[0]
            self.labels = dataset[1]
        else:
            self.inputs = dataset
            self.labels = None
        self.batch_size = batch_size
        self.shuffle = shuffle
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
                inputs = np.array([pad_array(i, size) for i in inputs])
            else:
                inputs = np.array([pad_sentence(i, size[0]) for i in inputs])
            inputs = torch.LongTensor(inputs)
            if self.labels is not None:
                labels = np.array([self.labels[i] for i in idx])
                labels = torch.LongTensor(labels)
                yield inputs, labels
            else:
                yield inputs


class TrainText(Dataset):
    def __init__(self, text_path, size=500):
        self.size = size
        self.array = isinstance(self.size, list)
        self.texts, self.labels, _ = pickle.load(open(text_path, "rb"))

    def __getitem__(self, index):
        if self.array:
            text = pad_array(self.texts[index], self.size)
        else:
            text = pad_sentence(self.texts[index], self.size)
        return text, self.labels[index]

    def __len__(self):
        return len(self.texts)


class TestText(Dataset):
    def __init__(self, text_path, size=500):
        self.size = size
        self.array = isinstance(self.size, list)
        self.texts, _ = pickle.load(open(text_path, "rb"))

    def __getitem__(self, index):
        if self.array:
            text = pad_array(self.texts[index], self.size)
        else:
            text = pad_sentence(self.texts[index], self.size)
        return text

    def __len__(self):
        return len(self.texts)


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

    def train(self, train_file, save_path, valid_file=None, train_cp=None, batch_size=128, learning_rate=1e-3, epochs=5, l2_lambda=1e-4, data_size=500):
        if not data_size:
            inputs, labels, _ = pickle.load(open(train_file, "rb"))
            train_data = TextDataLoader((inputs, labels), batch_size=batch_size, shuffle=True)
        else:
            train_data = DataLoader(TrainText(train_file, data_size), batch_size=batch_size, shuffle=True)
        if valid_file:
            valid_inputs, valid_labels, _ = pickle.load(open(valid_file, "rb"))
            if torch.cuda.is_available() and self.use_cuda:
                valid_inputs = valid_inputs.cuda()
                valid_labels = valid_labels.cuda()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_lambda)
        self.model.zero_grad()

        if train_cp:
            self.model.load_state_dict(torch.load(train_cp))

        for i in range(epochs):
            for j, data in enumerate(train_data):
                inputs, labels = Variable(data[0]), Variable(data[1])
                if torch.cuda.is_available() and self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                logits = self.model(inputs)
                loss = self.loss(logits, labels)
                loss.backward()
                optimizer.step()

                if j % 100 != 0:
                    continue
                print("Epoch %s, Batch %s :" %(i, j))
                _, predicted = torch.max(logits.data, 1)
                acu = (predicted == labels.data).sum() / len(labels)
                print("Train: Loss %.6f, Accuracy %.6f" %(loss, acu))
                cp_path = os.path.join(save_path, '%s_%s.cp' %(i, j))
                torch.save(self.model.state_dict(), cp_path)

                if not valid_file:
                    continue
                logits = self.model(valid_inputs)
                loss = self.loss(logits, valid_labels)
                _, predicted = torch.max(logits.data, 1)
                acu = (predicted == labels.data).sum() / len(labels)
                print("Test: Loss %.6f, Accuracy %.6f" % (loss, acu))

    def predict(self, predict_file, batch_size=128, data_size=500):
        if not data_size:
            inputs, _ = pickle.load(open(predict_file, "rb"))
            predict_data = TextDataLoader(inputs, batch_size=batch_size, shuffle=True)
        else:
            predict_data = DataLoader(TestText(predict_file, data_size), batch_size=batch_size)
        res = []
        for data in predict_data:
            inputs, labels = Variable(data[0]), Variable(data[1])
            if torch.cuda.is_available() and self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            logits = self.model(inputs)
            _, predicted = torch.max(logits.data, 1)
            res.append(predicted)
        return torch.cat(res)
