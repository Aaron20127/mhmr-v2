
import cv2
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
    def __init__(self, log_dir, config_path, save_obj=False, save_img=False):
        self.summary_id = 0

        # log dir
        self.log_dir = log_dir
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # copy config file
        shutil.copy(config_path, self.log_dir)

        # Summary Writer
        self.writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)

        # save obj
        if save_obj:
            self.save_obj_dir = os.path.join(self.log_dir, 'obj')
            os.makedirs(self.save_obj_dir, exist_ok=True)

        # save img
        if save_img:
            self.save_img_dir = os.path.join(self.log_dir, 'image')
            os.makedirs(self.save_img_dir, exist_ok=True)


    def update_summary_id(self, summary_id):
        self.summary_id = summary_id


    def scalar_summary_dict(self, tag_dict, prefix=''):
        """ Log a dict scalar variable. """
        for k, v in tag_dict.items():
            name = prefix + k
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
        """ add a image.
            img (tensor or narray, HxWxC)
        """
        self.writer.add_image(name, img,
                              global_step=self.summary_id,
                              dataformats='HWC')
        self.writer.flush()


    def add_text(self, name, text):
        """ add a text. """
        self.writer.add_text(name, text, self.summary_id)


    def add_mesh(self, name, vertices, faces, color=None):
        """ add a mesh.
            vertices (narray, BxNx3)
            faces (narray, BxNx3)
            color (narray, BxNx3)
        """
        self.writer.add_mesh(name, vertices=vertices, colors=color, faces=faces)
        self.writer.flush()


    def save_obj(self, name, vertices, faces):
        """ save .obj file to local dir.
            name ("*.obj") file name
        """
        obj_path = os.path.join(self.save_obj_dir, name)
        with open(obj_path, 'w') as fp:
            for v in vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


    def save_image(self, name, img):
        """ save .obj file to local dir.
            name ("*.png") file name
        """
        img_path = os.path.join(self.save_img_dir, name)
        cv2.imwrite(img_path, img)