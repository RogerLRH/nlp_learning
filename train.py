import argparse

import numpy as np
import tensorflow as tf

from textCNN import TextCNN
from textRNN import TextRNN
from textRCNN import TextRCNN
from data_treater import read_numbered


parser = argparse.ArgumentParser()

# parser.add_argument('--input_len', action='store', required=True, type=int, help='max sentence length')
# parser.add_argument('--num_class', action='store', required=True, type=int, type=int, help='number of labels')
parser.add_argument('--train_file', action='store', required=True, type=str, help='path of train data')
parser.add_argument('--ckpt_folder', action='store', default="checkpoints", type=str, help='path of checkpoint folder')
parser.add_argument('--train_cp', action='store', default=None, type=str, help='checkpoint to load for train, None by default')
parser.add_argument('--test_cp', action='store', required=True, type=str, help='checkpoint to load for test')
parser.add_argument('--epochs', action='store', default=5, type=int, help='epochs to circulate train data, 5 by default')
parser.add_argument('--hidden_size', action='store', default=100, type=int, help='LSTM hidden size, 100 by default')
parser.add_argument('--embed_size', action='store', default=100, type=int, help='embedding size, 100 by default')
parser.add_argument('--filter_sizes', action='store', default="1,2,3,4,5", type=str, help='sizes of filters, "1,2,3,4,5" by default')
parser.add_argument('--num_filter', action='store', default=256, type=int, help='number of filters for each size, 256 by default')
parser.add_argument('--learning_rate', action='store', default=0.001, type=float, help='learning rate, 0.001 by default')
parser.add_argument('--no_decay_step', action='store', default=1000, type=int, help='number of steps before decaying learning rate, 1000 by default')
parser.add_argument('--decay_rate', action='store', default=0.8, type=float, help='decay rate for learning rate, 0.8 by default')
parser.add_argument('--batch_size', action='store', default=128, type=int, help='batch size, 128 by default')
parser.add_argument('--pos_weight', action='store', default=1.0, type=float, help='weight of positive sample in sigmoid cross entropy, 1.0 by default')
parser.add_argument('--clip_gradient', action='store', default=5.0, type=float, help='clip gradients, 5.0 by default')
parser.add_argument('--multi_label', action='store', default=True, help='if one sample has multilabels, True by default')

args = parser.parse_args()

def calcul_loss(y, o):
    o[o <= 1e-7] = 1e-7
    o[o >= 1 - 1e-7] = 1 - 1e-7
    return np.mean(- y * np.log(o) - (1 - y) * np.log(1 - o))


def run():
    inputs, labels, voca_size = read_numbered(args.train_file)
    input_len = len(inputs[0])
    num_class = len(labels[0])
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    filter_sizes = [int(s) for s in args.filter_sizes.strip().split(",")]
    num_filter = args.num_filter
    learning_rate = args.learning_rate
    no_decay_step = args.no_decay_step
    decay_rate = args.decay_rate
    batch_size = args.batch_size
    pos_weight = args.pos_weight
    clip_gradient = args.clip_gradient
    multi_label = args.multi_label
    epochs = args.epochs
    train_cp = args.train_cp
    test_cp = args.test_cp
    ckpt_folder = args.ckpt_folder
    initializer = tf.random_normal_initializer(stddev=0.1)

    train_num = 140000
    train_x = inputs[:train_num]
    train_y = labels[:train_num]
    veri_x = inputs[train_num:]
    veri_y = labels[train_num:]
    test_x = veri_x[:1024]
    test_y = veri_y[:1024]

    clf = TextCNN(voca_size, input_len, num_class, embed_size, filter_sizes, num_filter, learning_rate, no_decay_step, decay_rate, batch_size, pos_weight, clip_gradient, initializer, multi_label)
    # clf = TextRNN(voca_size, input_len, hidden_size, num_class, embed_size, learning_rate, no_decay_step, decay_rate, batch_size, pos_weight, clip_gradient, initializer, multi_label)
    # clf = TextRCNN(voca_size, input_len, hidden_size, num_class, embed_size, learning_rate, no_decay_step, decay_rate, batch_size, pos_weight, initializer, multi_label, clip_gradient)
    clf.train(train_x, train_y, test_x, test_y, epochs=epochs, checkpoint=train_cp, save_path=ckpt_folder)
    _, probs = clf.test(veri_x, veri_y, test_cp)
    loss = calcul_loss(veri_y, probs)
    print(loss)

if __name__ == "__main__":
    run()
