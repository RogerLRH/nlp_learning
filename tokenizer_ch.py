# -*- coding: utf-8 -*-
import jieba

from base import is_english, is_chinese, load_lines2list, remove_stopword


stopfile = "CHstopwords.txt" # one word per line


def base_text2word(text):
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


class CHTokenizer(object):
    def __init__(self):
        self.stopwords = load_lines2list(stopfile)

    def get_words(self, text, mode="jieba"):
        if mode == "jieba":
            words = base_text2word(text.lower())
        elif mode == "bigram":
            words = text2bigram(text.lower())
        else:
            raise IOError("this mode is not defined")
        nostop = remove_stopword(words, self.stopwords)
        return nostop
