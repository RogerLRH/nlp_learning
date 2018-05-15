# -*- coding: utf-8 -*-
import pickle

from nlp_learning.const import NAME, TOKEN


def get_reverse_dict(dic):
    redic = {}
    for word, idx in dic.items():
        redic[idx] = word
    return redic


def save_dict(dic, dictfile):
    pickle.dump(dic, open(dictfile, "wb"))


def load_dict(dictfile):
    return pickle.load(open(dictfile, "rb"))


class Dictionary(object):
    """
    Build dictionary or turn text to index.

    Parameters
    ----------
    dictfile : string
        Path of dictionary file to load. Defaulf: None.

    Attributes
    ----------
    _dict_w2i : dict
        dict for word to index
    _min_index : int
        the min index can be used
    _dict_i2w : dict
        dict for index to word
    """
    def __init__(self, dictfile=None):
        self._dict_w2i = {
            NAME.PAD: 0,
            NAME.START: TOKEN.START,
            NAME.END: TOKEN.END,
            NAME.UNK: TOKEN.UNK}
        self._min_index = 1
        if dictfile:
            self._dict_w2i = load_dict(dictfile)
        self._dict_i2w = get_reverse_dict(self._dict_w2i)

    def build_dict(self, texts):
        for text in texts:
            self.add_dict_with_text(text)

    def add_dict_with_text(self, text):
        for word in text:
            if word in self._dict_w2i:
                continue
            self._reset_min_index()
            self._dict_w2i[word] = self._min_index
            self._dict_i2w[self._min_index] = word
            self._min_index += 1

    def _reset_min_index(self):
        while self._min_index in self._dict_i2w:
            self._min_index += 1

    def text2index(self, text):
        return [self._dict_w2i.get(word, TOKEN.UNK) for word in text]

    def texts2index(self, texts):
        return [self.text2index(text) for text in texts]

    def save(self, save_path):
        save_dict(self._dict_w2i, save_path)
