# -*- coding: utf-8 -*-
import argparse
import pickle

from nlp_learning.torch.text_classification.base_model import Treator
from nlp_learning.torch.text_classification.textCNN import TextCNN
from nlp_learning.torch.text_classification.textRNN import TextRNN, TextRCNN, TextRNNAttention, TextRNNAttentionWithSentence

from nlp_learning.torch.translation.base_model import Treator as TransTreator
from nlp_learning.torch.translation.seq2seq import Seq2Seq


parser = argparse.ArgumentParser()

parser.add_argument('--input_size', action='store', default=None, type=str, help='input size of one example')
parser.add_argument('--label_size', action='store', default=None, type=int, help='label size of one example')
parser.add_argument('--num_class', action='store', default=3, type=int, help='number of labels')
parser.add_argument('--train_file', action='store', required=True, type=str, help='path of train data')
parser.add_argument('--valid_file', action='store', default="", type=str, help='path of validate data')
parser.add_argument('--ckpt_folder', action='store', default="checkpoints", type=str, help='path of checkpoint folder')
parser.add_argument('--train_cp', action='store', default=None, type=str, help='checkpoint to load for training, None by default')
parser.add_argument('--epochs', action='store', default=5, type=int, help='epochs to circulate train data, 5 by default')
parser.add_argument('--hidden_size', action='store', default=100, type=int, help='LSTM hidden size, 100 by default')
parser.add_argument('--embed_size', action='store', default=100, type=int, help='embedding size, 100 by default')
parser.add_argument('--attn_size', action='store', default=100, type=int, help='attention size, 100 by default')
parser.add_argument('--filter_sizes', action='store', default="1,2,3", type=str, help='sizes of filters, "1,2,3" by default')
parser.add_argument('--num_filter', action='store', default=128, type=int, help='number of filters for each size, 128 by default')
parser.add_argument('--num_sampled', action='store', default=100, type=int, help='sampled num for softmax, 100 by default')
parser.add_argument('--learning_rate', action='store', default=0.001, type=float, help='learning rate, 0.001 by default')
parser.add_argument('--decay_step', action='store', default=1000, type=int, help='number of steps to make once decay rate, 1000 by default')
parser.add_argument('--decay_rate', action='store', default=0.8, type=float, help='decay rate for learning rate, 0.8 by default')
parser.add_argument('--batch_size', action='store', default=64, type=int, help='batch size, 64 by default')
parser.add_argument('--l2_lambda', action='store', default=0.0001, type=float, help='learning rate, 0.0001 by default')
parser.add_argument('--pos_weight', action='store', default=1.0, type=float, help='weight of positive sample in sigmoid cross entropy, 1.0 by default')
parser.add_argument('--clip_gradient', action='store', default=5.0, type=float, help='clip gradients, 5.0 by default')
parser.add_argument('--multi_label', action='store', default=False, help='if one sample has multilabels, False by default')
parser.add_argument('--use_cuda', action='store', default=False, help='if using cuda, False by default')

args = parser.parse_args()


def run():
    input_size = args.input_size
    if input_size:
        input_size = [int(s) for s in input_size.strip().split(",")]
        if len(input_size) == 1:
            input_size = input_size[0]
    label_size = args.label_size
    num_class = args.num_class
    train_file = args.train_file
    valid_file = args.valid_file
    save_path = args.ckpt_folder
    train_cp = args.train_cp
    epochs = args.epochs
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    attn_size = args.attn_size
    filter_sizes = [int(s) for s in args.filter_sizes.strip().split(",")]
    num_filter = args.num_filter
    num_sampled = args.num_sampled
    learning_rate = args.learning_rate
    decay_step = args.decay_step
    decay_rate = args.decay_rate
    batch_size = args.batch_size
    l2_ld = args.l2_lambda
    pos_weight = args.pos_weight
    clip_gradient = args.clip_gradient
    multi_label = args.multi_label
    use_cuda = args.use_cuda

    # _, _, voca_size = pickle.load(open(train_file, "rb"))
    #
    # model = TextCNN(voca_size, input_size, num_class, filter_sizes, num_filter, embed_size, use_cuda)
    # model = TextRNN(voca_size, num_class, hidden_size, embed_size, use_cuda)
    # model = TextRNNAttention(voca_size, num_class, hidden_size, embed_size, attn_size, use_cuda)
    # model = TextRNNAttentionWithSentence(voca_size, num_class, hidden_size, embed_size, attn_size, use_cuda)
    # model = TextRCNN(voca_size, input_size, num_class, hidden_size, embed_size, num_filter, use_cuda)

    # clf = Treator(model, multi_label, use_cuda)
    #
    # clf.train(train_file, save_path, valid_file, train_cp, batch_size, learning_rate, epochs, l2_ld, input_size)

    _, _, dict_size = pickle.load(open(train_file, "rb"))

    model = Seq2Seq(dict_size[0], dict_size[1], label_size, embed_size, hidden_size, attn_size, dropout_p=0.5, use_cuda=False)

    clf = TransTreator(model, use_cuda)

    clf.train(train_file, save_path, valid_file, train_cp, batch_size, learning_rate, epochs, input_size, label_size)


if __name__ == "__main__":
    run()
