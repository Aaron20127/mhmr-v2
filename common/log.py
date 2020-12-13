
import os
import time
import sys
import torch
import shutil
import tensorboardX

abspath = os.path.abspath(os.path.dirname(__file__))


class AverageLoss(object):
    def __init__(self):
        self.has_init = False
        self.count = 0

    def add(self, dict_in):
        if not self.has_init:
            self.count = 0
            self.has_init = True
            self.total_dict = dict_in

        else:
            for k, v in dict_in.items():
                self.total_dict[k] += v

        self.count += 1

    def get_average(self):
        average = {}

        for k, v in self.total_dict.items():
            average[k] = v / self.count

        return average

    def clear(self):
        self.has_init = False


class Logger(object):
    def __init__(self, opt, dst_dir):
        self.summary_id = 0

        # log dir
        self.log_dir = os.path.join(dst_dir, 'exp', opt.exp_name)
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # copy config file
        shutil.copy(dst_dir + '/config.py', self.log_dir)

        # Summary Writer
        self.writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)


    def update_summary_id(self, summary_id):
        self.summary_id = summary_id


    def scalar_summary_dict(self, tag_dict, prefix=''):
        """ Log a dict scalar variable. """
        for k, v in tag_dict.items():
            name = prefix + '_' + k
            self.writer.add_scalar(name, v, self.summary_id)
        self.writer.flush()


    def scalar_summary(self, tag, value):
        """ Log a scalar variable. """
        self.writer.add_scalar(tag, value, self.summary_id)
        self.writer.flush()


    def add_graph(self, model, input_to_model=None):
        """ add a graph. """
        self.writer.add_graph(model, input_to_model)
        self.writer.flush()


    def add_image(self, name, img):
        """ add a image. """
        self.writer.add_image(name, img, global_step=self.summary_id)
        self.writer.flush()
