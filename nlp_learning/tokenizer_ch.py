# -*- coding: utf-8 -*-
import jieba

from nlp_learning.base import is_english, is_chinese
from nlp_learning.base import load_lines2list
from nlp_learning.base import remove_stopword


base_stopfile = "CHstopwords.txt" # one word per line


def base_text2words(text):
    """
    Baseline method to get words, which should be chinese or english.

    Parameters
    ----------
    text : string

    Returns
    -------
    string list
    """
    words = []
    for w in jieba.cut(text):
        if len(w) > 1 and (is_chinese(w[0]) or is_english(w[0])):
            words.append(w)
    return words


# get only words combined by two chinese characters
def text2bigram(text):
    i, words = 0, []
    while i < len(text):
        if is_chinese(text[i]):
            if i + 1 < len(text) and is_chinese(text[i + 1]):
                words.append(text[i:i+2])
                i += 1
            else:
                i += 2
        else:
            i += 1
    return words


def text2words_jieba_direct(text):
    return jieba.lcut(text)


class CHTokenizer(object):
    """Tokenizer for chinese text.

    Parameters
    ----------
    stopfile : string
        Path of stopwords file.

    Attributes
    ----------
    stopwords : set
    """
    def __init__(self, stopfile=base_stopfile):
        self._stopwords = set(load_lines2list(stopfile))

    def get_words(self, text, mode="jieba"):
        text = text.lower()
        if mode == "jieba":
            words = base_text2words(text)
        elif mode == "bigram":
            words = text2bigram(text)
        elif mode == "jieba-direct":
            words = text2words_jieba_direct(text)
        else:
            raise IOError("this mode is not defined")
        nostop = remove_stopword(words, self._stopwords)
        return nostop
