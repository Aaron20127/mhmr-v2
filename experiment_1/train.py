
import os
import sys
import tqdm
import torch

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from .model import BaseNet, HmrLoss, ModelWithLoss
from common.checkpoint import load_model, save_model
from common.utils import show_net_para
from common.log import AverageLoss
from .opts import opt

from .dataloader import dataloader


class HMRTrainer(object):
    def __init__(self, logger):
        self.logger = logger
        self.start_epoch = 1
        self.log_id = 1
        self.val_id = 1

        self.build_model()
        self.create_data_loader()


    def build_model(self, opt):
        print('start building model ...')

        ### 1.object detection model
        model = BaseNet()
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

        if os.path.exists(opt.load_model):
            model, optimizer, self.start_epoch = \
              load_model(
                  model, opt.load_model,
                  optimizer, opt.resume,
                  opt.lr
              )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=opt.lr_scheduler_factor, patience=opt.lr_scheduler_patience,
            verbose=True, threshold=opt.lr_scheduler_threshold, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-10)

        self.model = model
        self.optimizer = optimizer
        # self.model_with_loss = \
        #     nn.DataParallel(ModelWithLoss(model, HmrLoss()), device_ids=opt.gpus_list)
        self.model_with_loss = ModelWithLoss(model, HmrLoss())

        show_net_para(model)
        print('finished build model.')


    def create_data_loader(self):
        print('start creating data loader ...')
        self.train_loader, self.val_loader, self.test_loader = dataloader()
        print('finished create data loader.')


    def run_val(self):
        """ run one epoch """
        data_loader = self.val_loader
        model_with_loss = self.model_with_loss
        model_with_loss.val()

        average_loss = AverageLoss()
        for i, batch in enumerate(data_loader):
            batch = [batch[k].to(opt.device, non_blocking=True) for k in batch.keys()]
            output, loss, loss_stats = model_with_loss(batch)

            average_loss.add(loss_stats)

        self.logger(average_loss.get_average(), self.val_id)


    def run_train_epoch(self):
        """ run one epoch """
        data_loader = self.train_loader
        model_with_loss = self.model_with_loss
        model_with_loss.train()

        average_loss = AverageLoss()

        for i, batch in enumerate(data_loader):
            ## forward
            batch = [batch[k].to(opt.device, non_blocking=True) for k in batch.keys()]
            output, loss, loss_stats = model_with_loss(batch)

            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)

            ## val
            if opt.val_iter_interval > 0 and \
               i % opt.val_interval == 0:
                self.run_val()
                self.model_with_loss.train()
                self.val_id += opt.val_iter_interval

            ## log
            average_loss.add(loss_stats)

            if opt.log_iter_interval > 0 and \
               i % opt.log_iter_interval == 0:
                self.logger(average_loss.get_average(), self.log_id)
                self.log_id += opt.log_iter_interval

            del output, loss, loss_stats


    def run(self):
        print('start training ...')

        start_epoch = self.start_epoch

        for epoch in tqdm.tqdm(range(start_epoch + 1, opt.num_epoch + 1)):

            if opt.train:
               self.run_train_epoch()

            if opt.val:
               self.run_val()

            if opt.train and \
               opt.save_epoch_interval > 0 and \
               epoch % opt.save_epoch_interval == 0:
                save_model(os.path.join(opt.save_dir, 'model_epoch_{}.pth'.format(epoch)),
                                epoch, self.model, self.optimizer)
