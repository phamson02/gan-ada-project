import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseGANTrainer
from utils import inf_loop, MetricTracker
from torch.autograd import Variable
import torch.autograd as autograd


class GANTrainer(BaseGANTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None):
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

        self.train_metrics = MetricTracker('g_loss', 'd_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

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

            # Loss measures generator's ability to fool the discriminator
            g_loss = self.criterion(self.model.discriminator(gen_imgs), valid)

            g_loss.backward()
            self.optimizer_G.step()

            # -----TRAIN DISCRIMINATOR-----
            self.optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.criterion(self.model.discriminator(real_imgs), valid)
            fake_loss = self.criterion(self.model.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()



            # Log loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss.item())
            self.train_metrics.update('d_loss', d_loss.item())

            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss.item(), d_loss.item()))
                # self.writer.add_image('input', make_grid(real_imgs.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log

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
            self.writer.add_image('fake', make_grid(fake_imgs.cpu(), nrow=8, normalize=True))

        # Add 32 real images to tensorboard
        real_imgs, _ = next(iter(self.data_loader))
        self.writer.set_step(epoch, 'valid')
        self.writer.add_image('real', make_grid(real_imgs.cpu()[:32], nrow=8, normalize=True))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

class WGANTrainer(BaseGANTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None):
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

        self.train_metrics = MetricTracker('g_loss', 'd_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

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

            # Loss measures generator's ability to fool the discriminator
            g_loss = self.criterion(self.model.discriminator(gen_imgs), valid)

            g_loss.backward()
            self.optimizer_G.step()

            # -----TRAIN DISCRIMINATOR-----
            self.optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.criterion(self.model.discriminator(real_imgs), valid)
            fake_loss = self.criterion(self.model.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            for p in self.model.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Log loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss.item())
            self.train_metrics.update('d_loss', d_loss.item())

            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss.item(), d_loss.item()))
                # self.writer.add_image('input', make_grid(real_imgs.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log

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
            self.writer.add_image('fake', make_grid(fake_imgs.cpu(), nrow=8, normalize=True))

        # Add 32 real images to tensorboard
        real_imgs, _ = next(iter(self.data_loader))
        self.writer.set_step(epoch, 'valid')
        self.writer.add_image('real', make_grid(real_imgs.cpu()[:32], nrow=8, normalize=True))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
class WGANGPTrainer(BaseGANTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, lr_scheduler_G=None, lr_scheduler_D=None, len_epoch=None, lambda_gp = 10):
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
        self.train_metrics = MetricTracker('g_loss', 'd_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

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
            # Loss measures generator's ability to fool the discriminator
            g_loss = -torch.mean(self.model.discriminator(gen_imgs))

            g_loss.backward()
            self.optimizer_G.step()

            # -----TRAIN DISCRIMINATOR-----
            self.optimizer_D.zero_grad()
            gradient_penalty = self.compute_gradient_penalty(self.model.discriminator, real_imgs.data, gen_imgs.data)
            # Measure discriminator's ability to classify real from generated samples
            real_loss = torch.mean(self.model.discriminator(real_imgs))
            fake_loss = torch.mean(self.model.discriminator(gen_imgs.detach()))
            d_loss = (- real_loss + fake_loss) / 2 + self.lamba_gp * gradient_penalty

            d_loss.backward()
            self.optimizer_D.step()

            # Log loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('g_loss', g_loss.item())
            self.train_metrics.update('d_loss', d_loss.item())

            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    g_loss.item(), d_loss.item()))
                # self.writer.add_image('input', make_grid(real_imgs.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self._valid_epoch(epoch)

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        return log

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
            self.writer.add_image('fake', make_grid(fake_imgs.cpu(), nrow=8, normalize=True))

        # Add 32 real images to tensorboard
        real_imgs, _ = next(iter(self.data_loader))
        self.writer.set_step(epoch, 'valid')
        self.writer.add_image('real', make_grid(real_imgs.cpu()[:32], nrow=8, normalize=True))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)