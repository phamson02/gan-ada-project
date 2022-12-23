import torch
import gc

from base import BaseGANTrainer


class LSGANTrainer(BaseGANTrainer):
    def __init__(self, model, criterion, optimizer_G, optimizer_D, config, device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch=None):
        super().__init__(model, criterion, optimizer_G, optimizer_D, config, device,
                         data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch)
        self.dis_a = config["trainer"]["a"] if config["trainer"]["a"] != "None" else 0
        self.dis_b = config["trainer"]["b"] if config["trainer"]["b"] != "None" else 1
        self.gen_c = config["trainer"]["c"] if config["trainer"]["c"] != "None" else 1

    def gen_loss(self, gen_imgs):
        disc_out = self.model.discriminator(gen_imgs).requires_grad_(True)
        g_loss = self.criterion(disc_out, torch.full([self.current_batch_size, 1], self.gen_c, dtype=torch.float32).to(
            self.device))

        return g_loss, disc_out.detach().cpu()

    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator(gen_imgs).requires_grad_(True)

        d_fake_loss = self.criterion(d_out_fake,
                                     torch.full([self.current_batch_size, 1], self.dis_a, dtype=torch.float32).to(
                                         self.device))

        return d_fake_loss, d_out_fake.detach().cpu()

    def d_real_loss(self, real_imgs):
        d_out_real = self.model.discriminator(real_imgs).requires_grad_(True)

        d_real_loss = self.criterion(d_out_real,
                                     torch.full([self.current_batch_size, 1], self.dis_b, dtype=torch.float32).to(
                                         self.device))

        return d_real_loss, d_out_real.detach().cpu()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.generator.train()
        self.model.discriminator.train()
        self.train_metrics.reset()

        for batch_idx, (real_imgs, _) in enumerate(self.data_loader):
            real_imgs = real_imgs.to(self.device)
            self.current_batch_size = real_imgs.shape[0]
            # -----TRAIN GENERATOR-----
            d_loss = self._train_G()

            # -----TRAIN DISCRIMINATOR-----
            g_loss, reals_out_D = self._train_D(real_imgs=real_imgs)

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

            # Log loss
            if self.writer:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss)
            self.train_metrics.update('d_loss', d_loss)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss, d_loss))

            del d_loss, g_loss
            gc.collect()

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log