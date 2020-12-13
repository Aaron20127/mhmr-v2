
import os
import sys
from tqdm import tqdm
import torch

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')

from model import BaseNet, HmrLoss, ModelWithLoss
from common.checkpoint import load_model, save_model
from common.utils import show_net_para
from common.log import AverageLoss
from common.data_parallel import DataParallel

from config import opt

from dataloader import all_loader


class HMRTrainer(object):
    def __init__(self, logger):
        self.logger = logger
        self.start_epoch = 0
        self.log_train_id = 0
        self.log_val_id = 0

        self.build_model()
        self.create_data_loader()
        self.set_device()

    def build_model(self):
        print('start building model ...')

        ### 1.object detection model
        model = BaseNet()
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

        if os.path.exists(opt.checkpoint_path):
            model, optimizer, self.start_epoch = \
              load_model(
                  model, opt.checkpoint_path,
                  optimizer, opt.resume
              )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer,
                                        mode='min',
                                        factor=opt.lr_scheduler_factor,
                                        patience=opt.lr_scheduler_patience,
                                        verbose=opt.lr_verbose,
                                        threshold=opt.lr_scheduler_threshold,
                                        threshold_mode='rel',
                                        cooldown=0, min_lr=0)

        self.model = model
        self.optimizer = optimizer
        # self.model_with_loss = \
        #     nn.DataParallel(ModelWithLoss(model, HmrLoss()), device_ids=opt.gpus_list)
        self.model_with_loss = ModelWithLoss(model, HmrLoss())

        show_net_para(model)
        print('finished build model.')

    def create_data_loader(self):
        print('start creating data loader ...')
        self.train_loader, self.val_loader, self.test_loader = all_loader()
        print('finished create data loader.')

    def set_device(self):
        if len(opt.gpus_list) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss,
                chunk_sizes=opt.chunk_sizes,
                device_ids=opt.gpus_list).to(opt.device)
        else:
            self.model_with_loss = self.model_with_loss.to(opt.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=opt.device, non_blocking=True)

    def val(self):
        """ run one epoch """
        average_loss = AverageLoss()

        with torch.no_grad():
            model_with_loss = self.model_with_loss
            if len(opt.gpus_list) > 1:
                model_with_loss = self.model_with_loss.module  # what this operation does?
            model_with_loss.eval()
            torch.cuda.empty_cache()

            for batch in self.val_loader:
                for k in batch.keys():
                    batch[k] = batch[k].to(opt.device, non_blocking=True)
                output, loss, loss_stats = self.model_with_loss(batch)

                average_loss.add(loss_stats)

        self.logger.scalar_summary_dict(average_loss.get_average(), 'val')
        # print('val %d/%d | loss %f' % (self.epoch, opt.num_epoch,
        #       average_loss.get_average()['loss']))


    def train_epoch(self):
        """ run one epoch """
        self.model_with_loss.train()
        average_loss = AverageLoss()
        len_data = len(self.train_loader)

        # log
        tqdm_loader = tqdm(self.train_loader, leave=True)
        tqdm_loader.set_description('train %d/%d' % (self.epoch, opt.num_epoch))

        for iter_id, batch in enumerate(tqdm_loader):
            iter_id += 1

            ## log id
            self.logger.update_summary_id((self.epoch-1) * len_data + iter_id)

            ## forward
            for k in batch.keys():
                batch[k] = batch[k].to(opt.device, non_blocking=True)
            output, loss, loss_stats = self.model_with_loss(batch)

            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)

            ## val
            if opt.val_iter_interval > 0 and \
               iter_id % opt.val_iter_interval == 0:
                tqdm_loader.set_description(' val  %d/%d' % (self.epoch, opt.num_epoch))
                self.val()
                tqdm_loader.set_description('train %d/%d' % (self.epoch, opt.num_epoch))

            ## log
            average_loss.add(loss_stats)

            if opt.train_iter_interval > 0 and \
               iter_id % opt.train_iter_interval == 0:

                # no average log
                # tqdm_loader.set_postfix(loss_stats)
                # self.logger.scalar_summary_dict(loss_stats, 'train')

                ## epoch average log
                # tqdm_loader.set_postfix(average_loss.get_average())
                self.logger.scalar_summary_dict(average_loss.get_average())

                ## train_iter_interval average log
                average_loss.clear()

            del output, loss, loss_stats


    def train(self):
        for epoch in range(self.start_epoch + 1, opt.num_epoch + 1):
            self.epoch = epoch

            # train
            self.train_epoch()

            # val
            if opt.val_epoch:
               self.val()

            if opt.save_epoch_interval > 0 and \
               epoch % opt.save_epoch_interval == 0:
                save_model(os.path.join(opt.logger.log_dir, 'model_epoch_{}.pth'.format(epoch)),
                            epoch, self.model, self.optimizer)



