import tensorflow as tf
import numpy as np
import argparse
from snow.common.output import Output
# from common.output import Output

reward_mat = np.array([
    0, 1, -1, 1,
    -1, 0, 1, 1,
    1, -1, 0, -1,
    -1, -1, 1, 0
])


def action_map(acts):
    return acts[0]*2+acts[1]


class MatrixGame:
    def __init__(self, args):
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        self.sig_size = args.sig_size
        self.hid_size = list(map(int, args.hid_size.split(','))) if len(args.hid_size) else []
        self.max_epoch_num = args.max_epoch_num
        self.deterministic = args.deterministic

        if args.ac_fn == 'tanh':
            self.fn = tf.nn.tanh
        elif args.ac_fn == 'relu':
            self.fn = tf.nn.relu
        elif args.ac_fn == 'sigmoid':
            self.fn = tf.nn.sigmoid
        elif args.ac_fn == 'elu':
            self.fn = tf.nn.elu
        else:
            raise ValueError

        self.indep = args.independent
        self.bs = args.bs
        self.lr = args.lr
        self.no_lr_decay = args.no_lr_decay
        self.com_sig = args.common_signal
        self.eval_interval = args.eval_interval
        self.eval_number = args.eval_number
        self.note = args.note
        self.restore_path = args.restore_path
        self.output = Output(args)
        self.wr = self.output.write

        self.sess = tf.Session()
        
    def run(self):
        self.setup_env()
        self.learn()

    def setup_env(self):

        self.lr_ = tf.placeholder(name='lr', dtype=tf.float32, shape=[])
        var_list = []

        with tf.variable_scope('row_player', reuse=tf.AUTO_REUSE):
            if self.indep:
                self.sig_row_ = tf.placeholder(name='sig_row', dtype=tf.float32, shape=[None, self.sig_size])
                pi_1_, vars = self.agt(self.sig_row_, '1')
                var_list.extend(vars[:])
                pi_2_, vars = self.agt(self.sig_row_, '2')
                var_list.extend(vars[:])
                self.pi_row_ = tf.reshape(tf.expand_dims(pi_1_, 2) * tf.expand_dims(pi_2_, 1), [-1, 4])
            else:
                self.sig_row_1_ = tf.placeholder(name='sig_row_1', dtype=tf.float32, shape=[None, 2 * self.sig_size])
                self.sig_row_2_ = tf.placeholder(name='sig_row_2', dtype=tf.float32, shape=[None, 2 * self.sig_size])
                pi_1_, vars = self.agt(self.sig_row_1_, 'row')
                var_list.extend(vars[:])
                pi_2_, _ = self.agt(self.sig_row_2_, 'row')
                self.pi_row_ = tf.reshape(tf.expand_dims(pi_1_, 2) * tf.expand_dims(pi_2_, 1), [-1, 4])

        with tf.variable_scope('col_player', reuse=tf.AUTO_REUSE):
            if self.indep:
                self.sig_col_ = tf.placeholder(name='sig_col', dtype=tf.float32, shape=[None, self.sig_size])
                pi_3_, vars = self.agt(self.sig_col_, '3')
                var_list.extend(vars[:])
                pi_4_, vars = self.agt(self.sig_col_, '4')
                var_list.extend(vars[:])
                self.pi_col_ = tf.reshape(tf.expand_dims(pi_3_, 2) * tf.expand_dims(pi_4_, 1), [-1, 4])
            else:
                self.sig_col_1_ = tf.placeholder(name='sig_col_1', dtype=tf.float32, shape=[None, 2 * self.sig_size])
                self.sig_col_2_ = tf.placeholder(name='sig_col_2', dtype=tf.float32, shape=[None, 2 * self.sig_size])
                pi_3_, vars = self.agt(self.sig_col_1_, 'col')
                var_list.extend(vars[:])
                pi_4_, _ = self.agt(self.sig_col_2_, 'col')
                self.pi_col_ = tf.reshape(tf.expand_dims(pi_3_, 2) * tf.expand_dims(pi_4_, 1), [-1, 4])

        self.pi_mat_ = tf.reshape(tf.expand_dims(self.pi_row_, 2) * tf.expand_dims(self.pi_col_, 1), [-1, 16])
        self.loss_row_ = -tf.reduce_mean(self.pi_mat_*reward_mat)
        # self.loss_row_ = tf.reduce_mean(tf.log(self.pi_mat_)*reward_mat)
        self.loss_col_ = -self.loss_row_

        # with tf.variable_scope('vf'):
        #     for var in var_list:
        #         tf.summary.histogram(var.name, var)
        with tf.variable_scope('others'):
            tf.summary.scalar('loss_row', self.loss_row_)
            tf.summary.histogram('pi_row', tf.reduce_mean(self.pi_row_, 0))
            tf.summary.histogram('pi_col', tf.reduce_mean(self.pi_col_, 0))

        self.train_op_row_ = tf.train.GradientDescentOptimizer(self.lr_).minimize(self.loss_row_,
                var_list=[v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='row_player')])
        # print([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='row_player')])
        self.train_op_col_ = tf.train.GradientDescentOptimizer(self.lr_).minimize(self.loss_col_,
                var_list=[v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='col_player')])
        saver = tf.train.Saver(max_to_keep=1)
        self.output.append_saver(saver, self.sess)

    def learn(self):

        self.sess.run(tf.global_variables_initializer())
        self.output.restore_model()
        self.train_writer_ = self.output.prepare_tsboard(self.sess.graph)
        self.merge_ = tf.summary.merge_all()

        eval_sig_row = np.random.normal(size=[self.eval_number, self.sig_size])
        if self.com_sig:
            eval_sig_col = eval_sig_row
        else:
            eval_sig_col = np.random.normal(size=[self.eval_number, self.sig_size])
            
        if self.indep:
            eval_feed_dict = {
                    self.sig_row_: eval_sig_row,
                    self.sig_col_: eval_sig_col,
                    self.lr_: 0}
        else:
            eval_feed_dict = {
                 self.sig_row_1_: np.concatenate([eval_sig_row, np.zeros_like(eval_sig_row), ], 1),
                 self.sig_row_2_: np.concatenate([np.zeros_like(eval_sig_row), eval_sig_row, ], 1),
                 self.sig_col_1_: np.concatenate([eval_sig_col, np.zeros_like(eval_sig_col), ], 1),
                 self.sig_col_2_: np.concatenate([np.zeros_like(eval_sig_col), eval_sig_col, ], 1),
                 self.lr_: 0}

        for ep in range(self.max_epoch_num):

            if self.no_lr_decay:
                lr = self.lr
            else:
                lr = (1 - ep / self.max_epoch_num) * self.lr

            sig_row = np.random.normal(size=[self.bs, self.sig_size])
            if self.com_sig:
                sig_col = sig_row
            else:
                sig_col = np.random.normal(size=[self.bs, self.sig_size])

            if self.indep:
                feed_dict = {self.sig_row_: sig_row,
                             self.sig_col_: sig_col,
                             self.lr_: lr}
            else:
                feed_dict = {self.sig_row_1_: np.concatenate([sig_row, np.zeros_like(sig_row), ], 1),
                             self.sig_row_2_: np.concatenate([np.zeros_like(sig_row), sig_row, ], 1),
                             self.sig_col_1_: np.concatenate([sig_col, np.zeros_like(sig_col), ], 1),
                             self.sig_col_2_: np.concatenate([np.zeros_like(sig_col), sig_col, ], 1),
                             self.lr_: lr}

            if ep % self.eval_interval == 0:

                pi_row, pi_col = self.sess.run([self.pi_row_, self.pi_col_], feed_dict=eval_feed_dict)

                if self.deterministic:
                    act_row, act_col = \
                    np.argmax(pi_row, axis=1)[:self.eval_number], \
                    np.argmax(pi_col, axis=1)[:self.eval_number]
                else:
                    act_row, act_col = \
                    [np.random.choice(4, 1, p=pi_row[i])[0] for i in range(len(pi_row))][:self.eval_number], \
                    [np.random.choice(4, 1, p=pi_col[i])[0] for i in range(len(pi_col))][:self.eval_number]
                eval_sig_row, eval_sig_col = eval_sig_row[:self.eval_number], eval_sig_col[:self.eval_number]
                for i in range(self.eval_number):
                    self.wr('[Eval:%i]|%s|%s|%s|%s'
                            % (ep, str(list(eval_sig_row[i])), str(list(eval_sig_col[i])),
                               str(act_row[i]), str(act_col[i])))

            summary, _, _, log_loss_row, log_loss_col = self.sess.run(
                [self.merge_, self.train_op_row_, self.train_op_col_, self.loss_row_, self.loss_col_],
                feed_dict=feed_dict)
            self.train_writer_.add_summary(summary, ep)

            if ep % (self.eval_interval//20) == 0:
                self.wr('[Train:%i]loss_row: %f' % (ep, log_loss_row))
                self.output.save_model()
                # self.wr('log_loss_col: %f' % log_loss_col)

    def agt(self, sig, name):
        last_out = sig
        var_list = []
        for i in range(len(self.hid_size)):
            last_out = self.fn(tf.layers.dense(last_out, self.hid_size[i], name="fc%i_%s" % (i + 1, name),
                                          kernel_initializer=tf.random_normal_initializer(mean=0)))
            var_list.append(last_out)
        pi_ = tf.layers.dense(last_out, 2, name='ac_%s' % name,
                                          kernel_initializer=tf.random_normal_initializer(mean=0))
        var_list.append(pi_)
        return tf.nn.softmax(pi_, dim=1), var_list
