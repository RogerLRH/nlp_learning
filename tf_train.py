# -*- coding: utf-8 -*-
import argparse
import pickle

from nlp_learning.tensorflow.text_classification.fastText import FastText
from nlp_learning.tensorflow.text_classification.textCNN import TextCNN
from nlp_learning.tensorflow.text_classification.textRNN import TextRNN, TextRNNAttention, TextRNNAttentionWithSentence
from nlp_learning.tensorflow.text_classification.textRCNN import TextRCNN


parser = argparse.ArgumentParser()

parser.add_argument('--input_size', action='store', default="500", type=str, help='input size of one example, can be "x" or "x,y". "500" by default')
parser.add_argument('--num_class', action='store', default=3, type=int, help='number of labels, 3 by default')
parser.add_argument('--train_file', action='store', required=True, type=str, help='path of train data')
parser.add_argument('--valid_file', action='store', default="", type=str, help='path of validate data, can not give, "" by default')
parser.add_argument('--ckpt_folder', action='store', default="checkpoints", type=str, help='path of checkpoint folder, checkpoints by default')
parser.add_argument('--train_cp', action='store', default=None, type=str, help='checkpoint to load for training, None by default')
parser.add_argument('--epochs', action='store', default=5, type=int, help='epochs to circulate train data, 5 by default')
parser.add_argument('--hidden_size', action='store', default=100, type=int, help='LSTM hidden size, 100 by default')
parser.add_argument('--embed_size', action='store', default=100, type=int, help='embedding size, 100 by default')
parser.add_argument('--filter_sizes', action='store', default="1,2,3", type=str, help='sizes of filters, "1,2,3" by default')
parser.add_argument('--num_filter', action='store', default=128, type=int, help='number of filters for each size, 128 by default')
parser.add_argument('--learning_rate', action='store', default=0.001, type=float, help='learning rate, 0.001 by default')
parser.add_argument('--decay_step', action='store', default=1000, type=int, help='number of steps to make once decay rate, 1000 by default')
parser.add_argument('--decay_rate', action='store', default=0.8, type=float, help='decay rate for learning rate, 0.8 by default')
parser.add_argument('--batch_size', action='store', default=64, type=int, help='batch size, 64 by default')
parser.add_argument('--l2_lambda', action='store', default=0.0001, type=float, help='learning rate, 0.0001 by default')
parser.add_argument('--pos_weight', action='store', default=1.0, type=float, help='weight of positive sample in sigmoid cross entropy, 1.0 by default')
parser.add_argument('--clip_gradient', action='store', default=5.0, type=float, help='clip gradients, 5.0 by default')
parser.add_argument('--multi_label', action='store', default=False, help='if one sample has multilabels, False by default')
parser.add_argument('--load_embed_only', action='store', default=False, help='if loaded checkpoint has only embedding, False by default')
parser.add_argument('--save_embed_only', action='store', default=False, help='if save only embedding, False by default')

args = parser.parse_args()


def run():
    input_len = [int(s) for s in args.input_size.strip().split(",")]
    if len(input_len) == 1:
        input_len = input_len[0]
    num_class = args.num_class
    train_file = args.train_file
    valid_file = args.valid_file
    save_path = args.ckpt_folder
    train_cp = args.train_cp
    epochs = args.epochs
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    filter_sizes = [int(s) for s in args.filter_sizes.strip().split(",")]
    num_filter = args.num_filter
    learning_rate = args.learning_rate
    decay_step = args.decay_step
    decay_rate = args.decay_rate
    batch_size = args.batch_size
    l2_ld = args.l2_lambda
    pos_weight = args.pos_weight
    clip_gradient = args.clip_gradient
    multi_label = args.multi_label
    load_embed_only = args.load_embed_only
    save_embed_only = args.save_embed_only

    _, _, dict_size = pickle.load(open(train_file, "rb"))
    # clf = FastText(dict_size, input_len, num_class, embed_size, l2_ld, pos_weight, multi_label, initial_size=.1)
    clf = TextCNN(dict_size, input_len, num_class, embed_size, filter_sizes, num_filter, l2_ld, pos_weight, multi_label, initial_size=.1)
    # clf = TextRNN(dict_size, input_len, num_class, hidden_size, embed_size, l2_ld, pos_weight, multi_label, initial_size=.1)
    # clf = TextRNNAttention(dict_size, input_len, num_class, hidden_size, embed_size, l2_ld, pos_weight, multi_label, initial_size=.1)
    # clf = TextRNNAttentionWithSentence(dict_size, input_len, num_class, hidden_size, embed_size, l2_ld, pos_weight, multi_label, initial_size=.1)
    # clf = TextRCNN(dict_size, input_len, num_class, hidden_size, embed_size, num_filter, l2_ld, pos_weight, multi_label, initial_size=.1)
    clf.train(train_file, valid_file, epochs, batch_size, learning_rate, decay_step, decay_rate, clip_gradient, checkpoint=train_cp, save_path=save_path, load_embed_only=load_embed_only, save_embed_only=save_embed_only)


if __name__ == "__main__":
    run()
