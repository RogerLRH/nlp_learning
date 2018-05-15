# -*- coding: utf-8 -*-
import re

from pattern3.en import lemma

from nlp_learning.base import is_english
from nlp_learning.base import load_lines2list
from nlp_learning.base import remove_stopword


stopfile = "ENstopwords.txt" # one word per line


def text2words(text):
    words = re.findall(r"[\w|']+", text)
    return [lemma(w) for w in words]


def only_english(words):
    return [w for w in words if w and is_english(w[0])]


class ENTokenizer(object):
    def __init__(self):
        self._stopwords = set(load_lines2list(stopfile))

    def get_words(self, text):
        words = only_english(text2words(text.lower()))
        nostop = remove_stopword(words, self._stopwords)
        return nostop
