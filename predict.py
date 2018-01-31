import argparse

import numpy as np
import tensorflow as tf

from textCNN import TextCNN
from data_treater import read_predict_numbered


parser = argparse.ArgumentParser()

# parser.add_argument('--input_len', action='store', required=True, type=int, help='max sentence length')
parser.add_argument('--num_class', action='store', default=6, type=int, help='number of labels')
parser.add_argument('--predict_file', action='store', required=True, type=str, help='path of predict data')
parser.add_argument('--ckpt_folder', action='store', default="checkpoints", type=str, help='path of checkpoint folder')
parser.add_argument('--checkpoint', action='store', default=None, type=str, help='checkpoint to load, None by default')
parser.add_argument('--epochs', action='store', default=5, type=int, help='epochs to circulate train data, 5 by default')
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


def run():
    inputs, voca_size = read_predict_numbered(args.predict_file)
    input_len = len(inputs[0])
    num_class = args.num_class
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
    checkpoint = args.checkpoint
    ckpt_folder = args.ckpt_folder
    initializer = tf.random_normal_initializer(stddev=0.1)

    clf = TextCNN(voca_size, input_len, num_class, embed_size, filter_sizes, num_filter, learning_rate, no_decay_step, decay_rate, batch_size, pos_weight, clip_gradient, initializer, multi_label)
    probs = clf.predict(inputs, checkpoint)
    

if __name__ == "__main__":
    run()
