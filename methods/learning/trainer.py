import os
from time import time
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import random
from tqdm import tqdm

from config import parse_args
from util import hex2rgb, AttrDict
from util.logger import logger
from util.pytorch import get_ckpt_path
from models import get_model_by_name
from components.dataset import get_dataset_by_name

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.config.output_min = 1.0
        self.config.output_max = 5.0

        # get dataset
        self.dataset = get_dataset_by_name(config.dataset)()
        self.config.input_dim = len(self.dataset[0]['item']['vec'])

        # get model
        self.model = get_model_by_name(config.model)(self.config)
        self.model = self.model.to(config.device)

        # get optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.lr)

        # setup logging
        if self.config.is_train:
            exclude = ['device']
            config_dict = self.config.__dict__.items()
            config_dict = {k: v for k, v in config_dict if k not in exclude}
            wandb.init(
                resume=config.run_name,
                project="faber",
                entity="jingyuny",
                config=config_dict,
                dir=config.log_dir,
                notes=config.notes
            )

    def _save_ckpt(self, step):
        ckpt_name = 'ckpt_{:08d}.pt'.format(step)
        ckpt_path = os.path.join(self.config.log_dir, ckpt_name)
        state_dict = {
            'step': step,
            'model': self.model.state_dict()
        }

        torch.save(state_dict, ckpt_path)
        logger.warn('Saved checkpoint at [{:s}]'.format(ckpt_path))

    def _load_ckpt(self, step=None):
        ckpt_path, step = get_ckpt_path(self.config.log_dir, step)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['step']
        else:
            logger.warn('Checkpoiont not found. Randomly initializing models.')
            return 0

    def _info2log(self, step, batch, info, k=8):
        visualized_info = {
            "loss": info["loss"],
            "rmse": info["rmse"],
            "acc": info["acc"]
        }

        if step % self.config.log_img_interval == 0:
            w, h, pad = 100, 5, 2
            item_h = h * 2 + pad
            img = np.zeros((item_h * k - pad, w, 3), dtype=float)

            def draw_bar(start_h, start_w, value, color):
                img[start_h:start_h+h, start_w:start_w+int(w*value), :] = color[None]

            for i in range(k):
                if np.abs(info['pred_labels'][i] - info['labels'][i]) < 0.25:
                    label_color = hex2rgb('4285f4')
                else:
                    label_color = hex2rgb('fbbc05')
                draw_bar(item_h * i, 0, info['pred_labels'][i] / 5.0, hex2rgb('#ffffff'))
                draw_bar(item_h * i + h, 0, info['labels'][i] / 5.0, label_color)

            for i in range(1, 5):
                start_w = int(np.asscalar(np.round(w / 5 * i)))
                img[:, start_w:start_w+1, :] *= 0.5

            visualized_info["img"] = img

        return visualized_info
        
    def _log_wandb(self, step, info, phase='train'):
        for k, v in info.items():
            if np.isscalar(v) or (hasattr(v, 'shape') and np.prod(v.shape) == 1):
                wandb.log({ '{}/{}'.format(phase, k): v }, step=step)
            else:
                imgs = [wandb.Image(v)]
                wandb.log({ '{}/{}'.format(phase, k): imgs }, step=step)
    
    def train(self):
        config = self.config

        # load checkpoint
        step = self._load_ckpt()

        logger.info("Start training at step #{:d}".format(step))
        progress_bar = tqdm(initial=step,
                            total=config.max_global_step,
                            desc=config.run_name)

        while step < config.max_global_step:
            train_info = {}
            
            # sample inputs
            logger.info("Sampling inputs.")
            batch = self.dataset.sample_batch(config.batch_size, mode='train')
            model_inputs, labels = self.model.process_input(batch)
            train_info.update({
                "labels": labels.detach().cpu().numpy()    
            })
            
            # compute embeddings for all samples
            pred_labels = self.model(*model_inputs)
            
            # compute loss and accuracy
            loss = torch.mean((pred_labels - labels) ** 2)
            rmse = torch.sqrt(loss)
            acc = torch.mean((torch.abs(pred_labels - labels) < 0.5).double())
            train_info.update({
                "pred_labels": pred_labels.detach().cpu().numpy(),
                "loss": loss.detach().cpu().numpy(),
                "rmse": rmse.detach().cpu().numpy(),
                "acc": acc.detach().cpu().numpy()
            })

            # update network
            logger.info("Update networks #{:8d}.".format(step))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # increment step
            step += 1

            # update progress bar
            progress_bar.update(1)

            # log training information occasionally
            if step % config.log_interval == 0:
                log = self._info2log(step, batch, train_info)
                self._log_wandb(step, log, phase='train')

            # run validation occasionally
            if step % config.val_interval == 0:
                val_info = {}

                val_batch = self.dataset.sample_batch(config.batch_size, mode='val')
                val_model_inputs, val_labels = self.model.process_input(val_batch)

                val_info.update({
                    "labels": val_labels.detach().cpu().numpy()    
                })

                # compute embeddings for all samples
                val_pred_labels = self.model(*val_model_inputs)
                
                # compute loss and accuracy
                val_loss = torch.mean((val_pred_labels - val_labels) ** 2)
                val_rmse = torch.sqrt(val_loss)
                val_acc = torch.mean((torch.abs(val_pred_labels - val_labels) < 0.5).double())
                val_info.update({
                    "pred_labels": val_pred_labels.detach().cpu().numpy(),
                    "loss": val_loss.detach().cpu().numpy(),
                    "rmse": val_rmse.detach().cpu().numpy(),
                    "acc": val_acc.detach().cpu().numpy()
                })

                log = self._info2log(step, batch, val_info)
                self._log_wandb(step, log, phase='val')

            # save checkpoint occasionally
            if step % config.ckpt_interval == 0:
                self._save_ckpt(step)

    def evaluate(self):
        raise NotImplementedError()

if __name__ == '__main__':
    config, _ = parse_args()
    config = AttrDict(vars(config))

    # setup log directory
    config.run_name = '{}.{}'.format(config.dataset, config.prefix)
    config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    logger.info('Create log directory: %s', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)

    # set global seed
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # set numpy options
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # device
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    # initialize and run trainer
    trainer = Trainer(config)
    if config.is_train:
        trainer.train()
        logger.info('Finished training.')
    else:
        trainer.evaluate()
        logger.info('Finished evaluation.')
