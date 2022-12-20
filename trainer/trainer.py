import numpy as np
import torch
from base import BaseGANTrainer
from utils import inf_loop, MetricTracker
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import gc
from model.loss import leastsquare_loss

class GANTrainer(BaseGANTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, optimizer_G, optimizer_D, config, device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch = None):
        super().__init__(model, criterion, optimizer_G, optimizer_D, config,device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch)
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
                    self.train_metrics.update('p', self.augment.p)
                    
                    del reals_out_D
                    gc.collect()
                    
                    self.lambda_t = list()  

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


class WGANTrainer(BaseGANTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer_G, optimizer_D, config, device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch = None):
        super().__init__(model, criterion, optimizer_G, optimizer_D, config,device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch)
        self.clip_value = self.config["trainer"]["clip"]
    def gen_loss(self, gen_imgs):
        disc_out = self.model.discriminator(gen_imgs).requires_grad_(True)
        #self.train_metrics.update('D(G(z))', torch.mean(nn.Sigmoid()(disc_out)))

        g_loss = -torch.mean(disc_out)

        return g_loss, disc_out.detach().cpu()    
    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator(gen_imgs.detach()).requires_grad_(True)

        d_fake_loss = torch.mean(d_out_fake)

        return d_fake_loss, d_out_fake.detach().cpu()

    def d_real_loss(self, real_imgs):
        d_out_real = self.model.discriminator(real_imgs).requires_grad_(True)

        d_real_loss = -torch.mean(d_out_real)

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
            # -----TRAIN DISCRIMINATOR-----
            d_loss, reals_out_D = self._train_D(real_imgs=real_imgs)
                        
            for param in self.model.discriminator.parameters():
                param.data.clamp_(-self.clip_value, self.clip_value)
            # -----TRAIN GENERATOR-----
            g_loss = self._train_G()


            self.iters += 1
            self.lambda_t.append(reals_out_D.sign().mean())
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(lambda_t=sum(self.lambda_t)/len(self.lambda_t),
                                          batch_size_D=reals_out_D.shape[0])
                    self.train_metrics.update('p', self.augment.p)
                    
                    del reals_out_D 
                    gc.collect()
                    
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


class WGANGPTrainer(BaseGANTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, optimizer_G, optimizer_D, config, device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch = None, lambda_gp = 10):
        super().__init__(model, criterion, optimizer_G, optimizer_D, config,device,
                 data_loader, augment, lr_scheduler_G, lr_scheduler_D, len_epoch)
        self.lambda_gp = lambda_gp
    def gen_loss(self, gen_imgs):
        disc_out = self.model.discriminator(gen_imgs).requires_grad_(True)
        #self.train_metrics.update('D(G(z))', torch.mean(nn.Sigmoid()(disc_out)))

        g_loss = -torch.mean(disc_out)

        return g_loss, disc_out.detach().cpu()    
    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator(gen_imgs.detach()).requires_grad_(True)

        d_fake_loss = torch.mean(d_out_fake)

        return d_fake_loss, d_out_fake.detach().cpu()

    def d_real_loss(self, real_imgs):
        d_out_real = self.model.discriminator(real_imgs).requires_grad_(True)

        d_real_loss = -torch.mean(d_out_real)

        return d_real_loss, d_out_real.detach().cpu()
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
        
        gradient_penalty = self.compute_gradient_penalty(self.model.discriminator, real_imgs, gen_imgs)

        d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gradient_penalty
        d_loss.backward()

        self.optimizer_D.step()

        ###LOG
        dx = (0.5 * torch.mean(nn.Sigmoid()(d_out_real)) + \
                                  0.5 * torch.mean(1 - nn.Sigmoid()(d_out_fake))).detach().cpu()
        self.train_metrics.update('D(x)', dx)
        self.train_metrics.update('d_out_real', d_out_real.numpy().mean())
        
        return d_loss.item(), d_out_real.detach()
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
            g_loss = self._train_G()

            # -----TRAIN DISCRIMINATOR-----
            d_loss, reals_out_D = self._train_D(real_imgs=real_imgs)

            self.iters += 1
            self.lambda_t.append(reals_out_D.sign().mean())
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(lambda_t=sum(self.lambda_t)/len(self.lambda_t),
                                          batch_size_D=reals_out_D.shape[0])
                    self.train_metrics.update('p', self.augment.p)
                    
                    del reals_out_D
                    gc.collect()

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
        g_loss = self.criterion(disc_out, torch.full([self.current_batch_size, 1], self.gen_c, dtype=torch.float32).to(self.device))

        return g_loss, disc_out.detach().cpu()

    def d_fake_loss(self, gen_imgs):
        d_out_fake = self.model.discriminator(gen_imgs).requires_grad_(True)

        d_fake_loss = self.criterion(nn.Sigmoid(d_out_fake), torch.full([self.current_batch_size, 1], self.dis_a, dtype=torch.float32).to(self.device))

        return d_fake_loss, d_out_fake.detach().cpu()

    def d_real_loss(self, real_imgs):
        d_out_real = self.model.discriminator(real_imgs).requires_grad_(True)

        d_real_loss = self.criterion(nn.Sigmoid(d_out_real), torch.full([self.current_batch_size, 1], self.dis_b, dtype=torch.float32).to(self.device))

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
            self.lambda_t.append(reals_out_D.sign().mean())
            # Update p value based on prediction of discriminator on real images
            if self.augment is not None and self.augment.name == "ADA":
                if self.iters % self.augment.integration_steps == 0:
                    self.augment.update_p(lambda_t=sum(self.lambda_t) / len(self.lambda_t),
                                          batch_size_D=reals_out_D.shape[0])
                    self.train_metrics.update('p', self.augment.p)

                    del reals_out_D
                    gc.collect()

                    self.lambda_t = list()

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

