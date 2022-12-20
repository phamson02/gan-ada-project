from logging import Logger
import torch
from torchvision.utils import make_grid
from abc import abstractmethod
from logger import TensorboardWriter, Wandb
from parse_config import ConfigParser
import torch.nn as nn
from utils import inf_loop, MetricTracker
import numpy as np
import wandb
import gc

class BaseGANTrainer:
    """
    Base class for all GAN trainers
    """
    def __init__(self, model, criterion, optimizer_G, optimizer_D, config, device,
                 data_loader, augment=None, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None):
        self.config: ConfigParser = config
        self.logger: Logger = config.get_logger('trainer', config['trainer']['verbosity'])


        self.model = model

        self.criterion = criterion

        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.augment = augment
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.valid = torch.ones(config["data_loader"]["args"]["batch_size"], 1).to(self.device)
        self.fake = torch.zeros(config["data_loader"]["args"]["batch_size"], 1).to(self.device)
        # setup visualization writer instance                
        if cfg_trainer['visual_tool'] in ['tensorboard', 'tensorboardX']:
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['visual_tool'])
        elif cfg_trainer['visual_tool'] == 'wandb':
            visual_config = {"Architecture": config['arch']['type'], "trainer": cfg_trainer["type"], "augment": config['augment']['type'] if config['augment']!={} else "None"}
            self.writer = Wandb(config['name'], cfg_trainer, self.logger, cfg_trainer['visual_tool'], visualize_config=visual_config)
        elif cfg_trainer['visual_tool'] == "None":
            self.writer = None
        else:
            raise ImportError("Visualization tool isn't exists, please refer to comment 1.* "
                              "to choose appropriate module")
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        self.train_metrics = MetricTracker('g_loss', 'd_loss', 'D(G(z))', 'D(x)', 'p', 'd_out_real', 'd_out_fake',
                                           writer=self.writer)
        self.iters = 0
        self.lambda_t = list()

    def _sample_noise(self, batch_size):
        return torch.randn(batch_size, self.model.latent_dim).to(self.device)

    def gen_loss(self, gen_imgs):
        disc_out = self.model.discriminator(gen_imgs).requires_grad_(True)
        g_loss = self.criterion(disc_out, self.valid[:self.current_batch_size])

        return g_loss, disc_out.detach().cpu()

    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator(gen_imgs).requires_grad_(True)

        d_fake_loss = self.criterion(d_out_fake, self.fake[:self.current_batch_size])

        return d_fake_loss, d_out_fake.detach().cpu()

    def d_real_loss(self, real_imgs):
        d_out_real = self.model.discriminator(real_imgs).requires_grad_(True)

        d_real_loss = self.criterion(d_out_real, self.valid[:self.current_batch_size])

        return d_real_loss, d_out_real.detach().cpu()
    def _train_D(self, real_imgs):
        """Function for training D, returning current loss and D's probability predictions on real samples"""
        self.optimizer_D.zero_grad()
        # Sample noise as generator input
        z = self._sample_noise(self.current_batch_size)
        # Generate a batch of images

        gen_imgs = self.model.generator(z)

        # Augment real and generated images
        if self.augment is not None:
            real_imgs = self.augment(real_imgs)
            gen_imgs = self.augment(gen_imgs)

        # Measure discriminator's ability to classify real from generated samples
        d_real_loss, d_out_real = self.d_real_loss(real_imgs)

        d_fake_loss, d_out_fake = self.d_fake_loss(gen_imgs=gen_imgs)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()

        self.optimizer_D.step()

        ###LOG
        d_x = (0.5 * torch.mean(nn.Sigmoid()(d_out_real)) +
               0.5 * torch.mean(1 - nn.Sigmoid()(d_out_fake))).detach().cpu().numpy()
        self.train_metrics.update('d_out_real', d_out_real.numpy().mean())
        self.train_metrics.update('D(x)', d_x)
        del d_x
        gc.collect()

        return d_loss.item(), d_out_real.detach()
    def _train_G(self, imgs=None):
        self.optimizer_G.zero_grad()
        z = self._sample_noise(self.current_batch_size)

        gen_imgs = self.model.generator(z)
        # Augment generated images
        if self.augment is not None:
            gen_imgs = self.augment(gen_imgs)
    
        g_loss, d_out_g = self.gen_loss(gen_imgs)
        g_loss.backward()

        self.optimizer_G.step()
        d_gz = torch.mean(nn.Sigmoid()(d_out_g)).detach().cpu().numpy()
        self.train_metrics.update('D(G(z))', d_gz)
        self.train_metrics.update('d_out_fake', d_out_g.numpy().mean())
        del d_gz, d_out_g
        gc.collect()

        return g_loss.item()

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

        if self.writer is not None:
            self.writer.writer.finish()

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
            :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir) + f'/{epoch}.pth'.zfill(4)
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_G']['type'] != self.config['optimizer_G']['type']:
            self.logger.warning("Warning: Optimizer_G type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

        if checkpoint['config']['optimizer_D']['type'] != self.config['optimizer_D']['type']:
            self.logger.warning("Warning: Optimizer_D type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        """
        if self.writer is not None:
            self.model.generator.eval()
            with torch.no_grad():
                noise = torch.randn(32, self.model.generator.latent_dim).to(self.device)
                fake_imgs = self.model.generator(noise)
                self.writer.set_step(epoch, 'valid')
                if self.writer.name == "tensorboard":
                    self.writer.add_image('fake', make_grid(fake_imgs, nrow=8, normalize=True))
                else:
                    images = wandb.Image(make_grid(fake_imgs[:32], nrow=8))
                    self.writer.log({'fake': images}, step=None)
                    
                    del images

                del noise, fake_imgs
                gc.collect()

            # Add 32 real images to tensorboard
            real_imgs, _ = next(iter(self.data_loader))
            self.writer.set_step(epoch, 'valid')
            if self.writer.name == "tensorboard":
                self.writer.add_image('real', make_grid(real_imgs[:32], nrow=8, normalize=True))
            else:
                images = wandb.Image(make_grid(real_imgs[:32], nrow=8))
                self.writer.log({'real': images}, step=None)
                del images

            del real_imgs
            gc.collect()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)