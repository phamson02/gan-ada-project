import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from torchvision.utils import make_grid
import wandb

from base import BaseGANTrainer
from utils.util import crop_image_by_part, copy_G_params, load_params


class FastGANTrainer(BaseGANTrainer):
    def __init__(self, model, criterion, optimizer_G, optimizer_D, config, device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch=None):
        super().__init__(model, criterion, optimizer_G, optimizer_D, config, device,
                         data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch)
        self.avg_param_G = copy_G_params(model.generator)
        self.fixed_noise = torch.randn(8, self.model.generator.latent_dim).to(self.device)
        self.save_interval = config["trainer"]["save_interval"]

    def init_lpips(self):
        import lpips
        self.percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    def gen_loss(self, gen_imgs):
        disc_out = self.model.discriminator(gen_imgs, label="fake").requires_grad_(True)
        g_loss = -disc_out.mean()

        return g_loss, disc_out.detach().cpu()

    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator([fi.detach() for fi in gen_imgs], label="fake").requires_grad_(True)
        d_fake_loss = F.relu(torch.rand_like(d_out_fake) * 0.2 + 0.8 + d_out_fake).mean()

        return d_fake_loss, d_out_fake.detach().cpu()

    def d_real_loss(self, real_imgs):
        part = random.randint(0, 3)
        d_out_real, [rec_all, rec_small, rec_part] = \
            self.model.discriminator(real_imgs, label="real", part=part)
        d_real_loss = F.relu(torch.rand_like(d_out_real) * 0.2 + 0.8 - d_out_real).mean() + \
                      self.percept(rec_all, F.interpolate(real_imgs, rec_all.shape[2])).sum() + \
                      self.percept(rec_small, F.interpolate(real_imgs, rec_small.shape[2])).sum() + \
                      self.percept(rec_part,
                                   F.interpolate(crop_image_by_part(real_imgs, part), rec_part.shape[2])).sum()

        return d_real_loss, d_out_real.detach().cpu(), rec_all.detach(), rec_small.detach(), rec_part.detach()

    def _train_D(self, real_imgs, gen_imgs):
        """Function for training D, returning current loss and D's probability predictions on real samples"""
        self.optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_real_loss, d_out_real, rec_all, rec_small, rec_part = self.d_real_loss(real_imgs)
        d_loss = d_real_loss.detach()
        d_real_loss.backward()

        d_fake_loss, d_out_fake = self.d_fake_loss(gen_imgs=gen_imgs)
        d_loss += d_fake_loss.detach()
        d_fake_loss.backward()

        self.optimizer_D.step()

        ###LOG
        d_x = (0.5 * torch.mean(nn.Sigmoid()(d_out_real)) +
               0.5 * torch.mean(1 - nn.Sigmoid()(d_out_fake))).detach().cpu().numpy()
        self.train_metrics.update('d_out_real', d_out_real.numpy().mean())
        self.train_metrics.update('D(x)', d_x)
        del d_x
        gc.collect()

        return d_loss.item(), d_out_real.detach(), rec_all, rec_small, rec_part

    def _train_G(self, gen_imgs):
        self.optimizer_G.zero_grad()

        g_loss, d_out_g = self.gen_loss(gen_imgs)
        g_loss.backward()

        self.optimizer_G.step()
        d_gz = torch.mean(nn.Sigmoid()(d_out_g)).detach().cpu().numpy()
        self.train_metrics.update('D(G(z))', d_gz)
        self.train_metrics.update('d_out_fake', d_out_g.numpy().mean())
        del d_gz, d_out_g
        gc.collect()

        return g_loss.item()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.generator.train()
        self.model.discriminator.train()
        self.train_metrics.reset()

        for batch_idx, (real_img, _) in enumerate(self.data_loader):
            real_img = real_img.to(self.device)

            self.current_batch_size = real_img.shape[0]
            # Fake images
            z = self._sample_noise(self.current_batch_size)
            gen_imgs = self.model.generator(z)

            # Augment generated and real images
            if self.augment is not None:
                gen_imgs = [self.augment(gen_img) for gen_img in gen_imgs]
                real_img = self.augment(real_img)

            # -----TRAIN DISCRIMINATOR-----
            d_loss, reals_out_D, rec_img_all, rec_img_small, rec_img_part = self._train_D(real_imgs=real_img,
                                                                                          gen_imgs=gen_imgs)

            self.iters += 1
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                self.augment.update_lambda(reals_out_D.sign().mean())
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(batch_size_D=reals_out_D.shape[0])
                    self.train_metrics.update('p', self.augment.p)

                    del reals_out_D
                    gc.collect()

                    self.augment.reset_lambda()

            # -----TRAIN GENERATOR-----
            g_loss = self._train_G(gen_imgs)

            ### EMA Generator update
            for p, avg_p in zip(self.model.generator.parameters(), self.avg_param_G):
                avg_p.mul_(0.999).add_(0.001 * p.data)

            if self.writer:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            if self.iters % 10 == 0:
                self.train_metrics.update('g_loss', -g_loss)
                self.train_metrics.update('d_loss', d_loss)
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    -g_loss, d_loss))

            if self.iters % (self.save_interval * 10) == 0:
                self._valid_iters(epoch)
                if self.writer != "None":
                    if self.writer.name == "tensorboard":
                        self.train_metrics.add_image('Image128', make_grid(torch.cat([
                            F.interpolate(real_img, 128),
                            rec_img_all, rec_img_small,
                            rec_img_part], dim=1).reshape(-1, 3, 128, 128), nrow=4, normalize=True))
                    else:
                        images = wandb.Image(make_grid(torch.cat([
                            F.interpolate(real_img, 128),
                            rec_img_all, rec_img_small,
                            rec_img_part], dim=1).reshape(-1, 3, 128, 128), nrow=4))
                        self.writer.log({'Image128': images}, step=None)

                        del images
                        del rec_img_all, rec_img_small, rec_img_part

            if self.iters % (self.save_interval * 50) == 0:
                self._save_checkpoint(epoch)

            del d_loss, g_loss
            gc.collect()

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log

    def train(self):
        """
        Full training logic
        """
        self.init_lpips()
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch == self.epochs:
                self._save_checkpoint(epoch)

        if self.writer is not None and self.writer.name == "wandb":
            self.writer.writer.finish()

    def _valid_iters(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        """
        if self.writer is not None:
            self.model.generator.eval()
            backup_para = copy_G_params(self.model.generator)
            load_params(self.model.generator, self.avg_param_G)

            with torch.no_grad():
                fake_imgs = self.model.generator(self.fixed_noise)
                self.writer.set_step(epoch, 'valid')
                if self.writer.name == "tensorboard":
                    self.writer.add_image('fake', make_grid(fake_imgs[0], nrow=4, normalize=True))
                else:
                    images = wandb.Image(make_grid(fake_imgs[0], nrow=4))
                    self.writer.log({'fake': images}, step=None)

                    del images

                del fake_imgs
                gc.collect()
            load_params(self.model.generator, backup_para)

            # Add 8 real images to tensorboard
            real_imgs, _ = next(iter(self.data_loader))
            self.writer.set_step(epoch, 'valid')
            if self.writer.name == "tensorboard":
                self.writer.add_image('real', make_grid(real_imgs[:8], nrow=4, normalize=True))
            else:
                images = wandb.Image(make_grid(real_imgs[:8], nrow=4))
                self.writer.log({'real': images}, step=None)
                del images

            del real_imgs
            gc.collect()

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
            'g_ema': self.avg_param_G,
            'state_dict': self.model.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'augment': self.augment.state_dict() if self.augment else None,
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
        self.avg_param_G = checkpoint['g_ema']
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

        # load augmentation state from checkpoint
        if checkpoint['config']['augment'] != self.config['augment']:
            self.logger.warning(
                "Warning: Augmentation type given in config file is different from that of checkpoint. "
                "Augmentation not being resumed.")
        if self.augment:
            self.augment.load_state_dict(checkpoint['augment'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))