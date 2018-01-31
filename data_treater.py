import numpy as np


def read_numbered(fname, sep="__label__"):
    inputs = []
    labels = []
    with open(fname) as f:
        line = f.readline().strip().split()
        sample_num = int(line[0])
        voca_size = int(line[1])
        while True:
            line = f.readline().strip()
            if not line:
                break
            line = line.split(sep)
            inputs.append(np.array([int(w) for w in line[0].split()]))
            labels.append(np.array([int(w) for w in line[1].split()]))
    if len(inputs) != sample_num:
        print("warnning: the sample number is wrong, please check the train data file")
    return np.array(inputs), np.array(labels), voca_size

def read_predict_numbered(fname):
    inputs = []
    with open(fname) as f:
        line = f.readline().strip().split()
        sample_num = int(line[0])
        voca_size = int(line[1])
        while True:
            line = f.readline().strip()
            if not line:
                break
            inputs.append(np.array([int(w) for w in line.split()]))
    if len(inputs) != sample_num:
        print("warnning: the sample number is wrong, please check the train data file")
    return np.array(inputs), voca_size
