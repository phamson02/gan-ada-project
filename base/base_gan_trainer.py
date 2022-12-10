from logging import Logger
import torch
from torchvision.utils import make_grid
from abc import abstractmethod
from logger import TensorboardWriter, Wandb
from parse_config import ConfigParser
import wandb

class BaseGANTrainer:
    """
    Base class for all GAN trainers
    """
    def __init__(self, model, criterion, optimizer_G, optimizer_D, config):
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


        # setup visualization writer instance
        if cfg_trainer['visual_tool'] in ['tensorboard', 'tensorboardX']:
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['visual_tool'])
        elif cfg_trainer['visual_tool'] == 'wandb':
            visual_config = {"Architecture": config['arch']['type'], "trainer": cfg_trainer["type"], "augment": config['augment']['type']}
            self.writer = Wandb(cfg_trainer, self.logger, cfg_trainer['visual_tool'], visualize_config=visual_config)
        elif cfg_trainer['visual_tool'] == "None":
            self.writer = None
        else:
            raise ImportError("Visualization tool isn't exists, please refer to comment 1.* "
                              "to choose appropriate module")

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

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
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
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
        self.model.generator.eval()
        with torch.no_grad():
            noise = torch.randn(32, self.model.generator.latent_dim).to(self.device)
            fake_imgs = self.model.generator(noise)
            self.writer.set_step(epoch, 'valid')
            if self.writer.name == "tensorboard":
                self.writer.add_image('fake', make_grid(fake_imgs.cpu(), nrow=8, normalize=True))
            else:
                images = wandb.Image(make_grid(fake_imgs.cpu()[:32], nrow=8))
                self.writer.log({'fake': images})
        # Add 32 real images to tensorboard
        real_imgs, _ = next(iter(self.data_loader))
        self.writer.set_step(epoch, 'valid')
        if self.writer.name == "tensorboard":
            self.writer.add_image('real', make_grid(real_imgs.cpu()[:32], nrow=8, normalize=True))
        else:
            images = wandb.Image(make_grid(real_imgs.cpu()[:32], nrow=8))
            self.writer.log({'real': images})

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)