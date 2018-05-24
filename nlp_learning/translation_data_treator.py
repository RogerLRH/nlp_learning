# -*- coding: utf-8 -*-
import pickle

from nlp_learning.text2index import Dictionary


def eng_fra_tokenizer(sentence):
    """
    tokenizer for sentences from fra.txt, which is from http://www.manythings.org/anki

    Parameters
    ----------
    text : string
        english sentence or french sentence
    Returns
    -------
    string list
        sentence in wordlist
    """
    wordlist = sentence.lower().split()
    new = []
    for word in wordlist:
        newwords = [word]
        if len(word) > 1 and word[-1] in ("!", "?", ".", ",", ":", '"'):
            newwords = [word[:-1], word[-1]]
        if len(word) > 2 and "'" in word:
            part = newwords[0].split("'")
            newwords = [part[0], "'", part[1]] + newwords[1:]
        if len(word) > 2 and "-" in word:
            newwords = " - ".join(newwords[0].split("-")).split() + newwords[1:]
        if len(word) > 1 and word[0] == '"':
            newwords = [newwords[0][0], newwords[0][1:]] + newwords[1:]
        new.extend(newwords)
    return new


def pretreat_corpus(corpus_file, pretreated_file):
    """
    Pretreat corpus file of english and another language, store in a pkl file.
    corpus_file is from http://www.manythings.org/anki
    """
    engs, others = [], []
    with open(corpus_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        engs.append(eng_fra_tokenizer(line[0]))
        others.append(eng_fra_tokenizer(line[1]))
    pickle.dump((engs, others), open(pretreated_file, "wb"))


def build_dict_with_pretreated_corpus(pretreated_file, eng_dict_file, other_dict_file):
    engs, others = pickle.load(open(pretreated_file, "rb"))
    eng_dict = Dictionary()
    other_dict = Dictionary()
    eng_dict.build_dict(engs)
    other_dict.build_dict(others)
    eng_dict.save(eng_dict_file)
    other_dict.save(other_dict_file)


def build_train_file(pretreated_file, train_file, eng_dict_file, other_dict_file):
    engs, others = pickle.load(open(pretreated_file, "rb"))
    eng_dict = Dictionary(eng_dict_file)
    other_dict = Dictionary(other_dict_file)
    eng_index = eng_dict.texts2index(engs, with_end=True)
    other_index = other_dict.texts2index(others, with_end=True)
    size = [eng_dict.get_dict_size(), other_dict.get_dict_size()]
    pickle.dump((eng_index, other_index, size), open(train_file, "wb"))
