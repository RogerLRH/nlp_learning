import tensorflow as tf

from base_model import BaseModel


class FastText(BaseModel):
    def __init__(self, voca_size, input_len, hidden_size, num_class, embed_size=100, learning_rate=1e-3, decay_step=1000, decay_rate=0.8, batch_size=128, pos_weight=1, clip_gradient=5.0, initializer=tf.random_normal_initializer(stddev=0.1), multi_label=False):
        # set hyperparamter
        self.voca_size = voca_size
        self.embed_size = embed_size

        super(FastText, self).__init__(input_len=input_len, num_class=num_class, learning_rate=learning_rate, decay_step=decay_step, decay_rate=decay_rate, batch_size=batch_size, pos_weight=pos_weight, clip_gradient=clip_gradient, initializer=initializer, multi_label=multi_label)


    def init_weights(self):
        # define all weights here
        with tf.name_scope("embed"):
            self.embedding = tf.get_variable("embedding", shape=[self.voca_size, self.embed_size], initializer=self.initializer)

        with tf.name_scope("full"):
            self.W_project = tf.get_variable("W_project", shape=[self.embed_size, self.num_class], initializer=self.initializer)
            self.b_project = tf.get_variable("b_project", shape=[self.num_class])

    def core(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # emebedding
        embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input)  # [None, self.input_len, self.embed_size]

        # average of vectors
        self.embeds = tf.reduce_mean(embedded_sentence, axis=1)

        # fc
        logits = tf.matmul(self.embeds, self.W_project) + self.b_project
        return logits
