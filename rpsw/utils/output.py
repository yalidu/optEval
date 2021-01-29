import time
import sys
from functools import partial
import os
from time import gmtime, strftime
import tensorflow as tf
import shutil
from os import listdir
from os.path import isfile, join


class Output:

    def __init__(self, args):

        # 准备terminal内的动态打印和静态打印
        self.sprint, self.dprint = init_print()

        # 准备输出的文件
        output_path, note = args.output_path, args.note
        folder = '%s/' % note
        self.output_path = output_path+folder
        self.output_file = self.output_path + 'results.txt'
        self.debug_file = self.output_path + 'debug_log.txt'
        self.restore_path = args.restore_path

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # if not os.path.exists(self.output_path+'log/'):
        #     os.mkdir(self.output_path + 'log/')

        if os.path.isfile(self.output_path+'.finish'):
            os.remove(self.output_path+'.finish')

        with open(self.output_file, 'w') as _:
            pass

        self.f = open(self.output_file, 'a', 1)
        self.debug_f = open(self.debug_file, 'a', 1)
        self.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        self.write(str(args))

    def update_restore_path(self, path):
        self.restore_path = path

    def write(self, message):

        # 将message输入到指定文件内
        self.sprint(message)
        self.f.write(message+'\n')

    def debug_write(self, message, stdout=False):

        # 将message输入到指定文件内
        if stdout:
            self.sprint(message)
        self.debug_f.write(message+'\n')

    def append_saver(self, saver, sess):

        self.saver = saver
        self.sess = sess

    def save_model(self, path=None):

        if not path:
            path = self.output_path
        save_path = self.saver.save(self.sess, path + 'model.ckpt')
        self.write('[Save] Model saved in %s' % save_path)
        return save_path

    def auto_save(self, save_interval, epoch):
        if epoch and not epoch % save_interval:
            self.save_model(self.output_path+'%i/' % epoch)

    def restore_model(self, model_name='model.ckpt'):

        # model_file = tf.train.latest_checkpoint(self.restore_path)  # changed
        if self.restore_path:
            self.saver.restore(self.sess, self.restore_path + model_name)
            self.write('[Restore] Model restored from %s.' % self.restore_path)
        else:
            self.write('[Init] Initialized from random.')
        return self.restore_path

    def prepare_tsboard(self, graph):
        filelist = [f for f in os.listdir(self.output_path + '/')]
        for f in filelist:
            if f.startswith('events.out.tfevents.'):
                os.remove(self.output_path+f)
        self.train_writer_ = tf.summary.FileWriter(self.output_path, graph)
        return self.train_writer_

    def close(self):

        with open(self.output_path+'.finish', 'w'):
            pass


def init_print(silence=False):
    start_time = time.time()
    sprint = partial(static_print, start_time=start_time)
    dprint = partial(dynamic_print, start_time=start_time, silence=silence)
    return sprint, dprint


def static_print(messages, start_time, silence=False, decorator=None):

    assert type(messages) == str or type(messages) == list
    assert not decorator or decorator == 'both' or decorator == 'before' or decorator == 'after'

    if not silence:
        if type(messages) == str:
            messages = [messages]

        if decorator == 'before' or decorator == 'both':
            print('-'*50)
        for message in messages:
            sys.stdout.write(' ' * 50 + '\r')
            sys.stdout.flush()
            print(message + '  [ %is ]' % (time.time() - start_time))
        if decorator == 'after' or decorator == 'both':
            print('-'*50)


def dynamic_print(message, start_time, silence=False):

    assert type(message) == str

    if not silence:
        sys.stdout.write(' ' * 110 + '\r')
        sys.stdout.flush()
        sys.stdout.write(message + '  [ %is ]' % (time.time() - start_time) + '\r')
        sys.stdout.flush()
