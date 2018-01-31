# -*- coding: utf-8 -*-
import os

import numpy as np


def is_chinese(chara):
    if u'\u4e00' <= chara <= u'\u9fa5':
        return True
    return False


def is_english(chara):
    if u'\u0061' <= chara <= u'\u007a' or u'\u0041' <= chara <= u'\u005a':
        return True
    return False


def is_number(chara):
    if u'\u0030' <= chara <= u'\u0039':
        return True
    return False


def label2onehot(labels, num_class):
    return np.eye(num_class)[labels]


def load_lines2list(filename):
    res = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.strip())
    return res


def remove_stopword(words, stopwords):
    return [w for w in words if w not in stopwords]


class BuildDict(object):
    # build dictionary and turn word list into index text2index
    def __init__(self):
        self.dict = {}

    def text2index(self, text, build=True):
        # 词列表转化为index列表
        return [self.word2index(word, build) for word in text]

    def word2index(self, word, build):
        # 构造词典，返回对应index
        if word not in self.dict and build:
            self.dict[word] = len(self.dict) + 1
        return self.dict.get(word, 0)


def get_reverse_dict(dic):
    redic = {}
    for word, idx in dic.items():
        redic[idx] = word
    return redic


def save_dict(dic, dictfile):
    with open(dictfile, "w") as f:
        for word, idx in dic:
            f.write(str(idx) + " " + word + "\n")


def load_dict(dictfile):
    d = {}
    with open(dictfile) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            d[line[1]] = int(line[0])
    return d


def get_text_from_file(fname, at_type):
    with open(fname) as f:
        if at_type == "text":
            text = "".join(f.readlines())
        if at_type == "words":
            text = f.readline().strip().split()
    return text


# Get all texts. Each folder's name is the category.
def get_text_from_folders(path, at_type="words"):
    for (_, dirnames, _) in os.walk(path):
        folders = dirnames
        break
    for category in folders:
        prepath = os.path.join(path, category)
        for (_, _, fnames) in os.walk(prepath):
            for fname in fnames:
                fname = os.path.join(prepath, fname)
                text = get_text_from_file(fname, at_type)
                yield category, text
