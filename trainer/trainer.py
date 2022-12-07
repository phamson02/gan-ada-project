import numpy as np
import torch
from base import BaseGANTrainer
from utils import inf_loop, MetricTracker
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn


class GANTrainer(BaseGANTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, augment=None, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer_G, optimizer_D, config)
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

        self.train_metrics = MetricTracker('g_loss', 'd_loss', 'D(G(z))', 'D(x)',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.iters = 0
        self.lambda_t = list()

    def _sample_noise(self, batch_size):
        return torch.randn(batch_size, self.model.generator.latent_dim).to(self.device)

    def gen_loss(self, gen_imgs):
        disc_out = self.model.discriminator(gen_imgs).requires_grad_(True)

        self.train_metrics.update('D(G(z))', torch.mean(nn.Sigmoid()(disc_out)))

        g_loss = self.criterion(disc_out, self.valid[:self.current_batch_size])

        return g_loss

    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator(gen_imgs).requires_grad_(True)

        d_fake_loss = self.criterion(d_out_fake, self.fake[:self.current_batch_size])

        return d_fake_loss, d_out_fake

    def d_real_loss(self, real_imgs):
        d_out_real = self.model.discriminator(real_imgs).requires_grad_(True)

        d_real_loss = self.criterion(d_out_real, self.valid[:self.current_batch_size])

        return d_real_loss, d_out_real

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
        self.train_metrics.update('D(x)', 0.5 * torch.mean(nn.Sigmoid()(d_out_real)) + \
                                  0.5 * torch.mean(1 - nn.Sigmoid()(d_out_fake)))
        return d_loss.item(), d_out_real.detach()

    def _train_G(self):
        self.optimizer_G.zero_grad()
        z = self._sample_noise(self.current_batch_size)

        gen_imgs = self.model.generator(z)
        # Augment generated images
        if self.augment is not None:
            gen_imgs = self.augment(gen_imgs)

        g_loss = self.gen_loss(gen_imgs)
        g_loss.backward()

        self.optimizer_G.step()
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

        for batch_idx, (real_imgs, _) in enumerate(self.data_loader):
            real_imgs = real_imgs.to(self.device)
            self.current_batch_size = real_imgs.shape[0]
            # -----TRAIN GENERATOR-----
            d_loss = self._train_G()

            # -----TRAIN DISCRIMINATOR-----
            g_loss, reals_out_D = self._train_D(real_imgs=real_imgs)

            self.iters += 1
            self.lambda_t.append(reals_out_D.sign().mean())
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(lambda_t=sum(self.lambda_t)/len(self.lambda_t),
                                          batch_size_D=reals_out_D.shape[0])
                    self.lambda_t = list()

            # Log loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss)
            self.train_metrics.update('d_loss', d_loss)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss, d_loss))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log


class WGANTrainer(BaseGANTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, augment=None, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
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

        self.train_metrics = MetricTracker('g_loss', 'd_loss', *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)
        self.augment = augment
        self.iters = 0
        self.lambda_t = list()

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

            # -----TRAIN GENERATOR-----
            # Adversarial ground truths
            valid = torch.ones(real_imgs.size(0), 1).to(self.device)
            fake = torch.zeros(real_imgs.size(0), 1).to(self.device)

            self.optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(real_imgs.size(0), self.model.generator.latent_dim).to(self.device)

            # Generate a batch of images
            gen_imgs = self.model.generator(z)

            # augment real and generated images
            if self.augment is not None:
                real_imgs = self.augment(real_imgs)
                gen_imgs = self.augment(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = self.criterion(self.model.discriminator(gen_imgs), valid)

            g_loss.backward()
            self.optimizer_G.step()

            # -----TRAIN DISCRIMINATOR-----
            self.optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_logits = self.model.discriminator(real_imgs)
            real_loss = self.criterion(real_logits, valid)
            fake_loss = self.criterion(self.model.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            for p in self.model.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            reals_out_D = torch.randn(4,2,1)
            self.iters += 1
            self.lambda_t.append(reals_out_D.sign().mean())
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(lambda_t=sum(self.lambda_t) / len(self.lambda_t),
                                          batch_size_D=reals_out_D.shape[0])
                    self.lambda_t = list()

            # Log loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss.item())
            self.train_metrics.update('d_loss', d_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss.item(), d_loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log


class WGANGPTrainer(BaseGANTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, augment=None, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None, lambda_gp=10):
        super().__init__(model, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
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
        self.lamba_gp = lambda_gp
        self.augment = augment
        self.train_metrics = MetricTracker('g_loss', 'd_loss', *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)
        self.iters = 0
        self.lambda_t = list()

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        cuda = True if torch.cuda.is_available() else False

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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
            cuda = True if torch.cuda.is_available() else False

            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

            real_imgs = real_imgs.to(self.device)
            real_imgs = Variable(real_imgs.type(Tensor))

            # -----TRAIN GENERATOR-----
            # Adversarial ground truths
            valid = torch.ones(real_imgs.size(0), 1).to(self.device)
            fake = torch.zeros(real_imgs.size(0), 1).to(self.device)

            self.optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(real_imgs.size(0), self.model.generator.latent_dim).to(self.device)

            # Generate a batch of images
            gen_imgs = self.model.generator(z)

            if self.augment is not None:
                real_imgs = self.augment(real_imgs)
                gen_imgs = self.augment(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = -torch.mean(self.model.discriminator(gen_imgs))

            g_loss.backward()
            self.optimizer_G.step()

            # -----TRAIN DISCRIMINATOR-----
            self.optimizer_D.zero_grad()
            gradient_penalty = self.compute_gradient_penalty(self.model.discriminator, real_imgs.data, gen_imgs.data)
            # Measure discriminator's ability to classify real from generated samples
            real_logits = self.model.discriminator(real_imgs)

            real_loss = torch.mean(real_logits)
            fake_loss = torch.mean(self.model.discriminator(gen_imgs.detach()))
            d_loss = (- real_loss + fake_loss) / 2 + self.lamba_gp * gradient_penalty

            d_loss.backward()
            self.optimizer_D.step()

            reals_out_D = torch.randn(4,2,1)
            self.iters += 1
            self.lambda_t.append(reals_out_D.sign().mean())
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(lambda_t=sum(self.lambda_t) / len(self.lambda_t),
                                          batch_size_D=reals_out_D.shape[0])
                    self.lambda_t = list()

            # Log loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss.item())
            self.train_metrics.update('d_loss', d_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss.item(), d_loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log