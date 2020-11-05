import torch
import torchvision
import datetime
import pytorch_msssim

from torch import nn
from model import srgan
from torch import optim
from torchvision import models
from collections import namedtuple
from model.dataset import SRDataset
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from math import log10
from statistics import mean
from tqdm import tqdm

#-------------------------------------------------------------------------------
class SRGANTrainer():

    def __init__(
        self, 
        train_data,
        test_data,
        logdir = "./log/",
        num_iterations = 2e5,
        num_workers = 4,
        batch_size = 64
    ):
        """
        """

        # Training Parameters:
        self.LR = 0.001
        self.device = "cpu"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_factor = 1e-3
        
        # Extract the convolutional layers of the VGG19 network used to compute 
        # the VGG content loss.
        self.vgg = models.vgg19(pretrained = True).to(self.device)
        self.vgg = self.vgg[:19]
        self.vgg.eval()

        # Create the training and test dataset and loaders:
        self.train_dataset = SRDataset(train_data, mode='train', scale_factor=4)
        self.test_dataset = SRDataset(test_data, mode='test', scale_factor=4)

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size, 
            shuffle = True,
            num_workers = self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size = self.batch_size, 
            shuffle = False,
            num_workers = self.num_workers
        )

        # Create tensorboard writer to log trainings:
        self.logger = tensorboard.SummaryWriter(logdir)

        # Create the GAN network:
        self.generator     = srgan.Generator()
        self.discriminator = srgan.Discriminator()

        # Create optimizers for each network:
        self.gen_optimizer = self.__getoptimizer__(self.generator)
        self.dis_optimizer = self.__getoptimizer__(self.discriminator)

        # Flag the networks to train mode:
        self.generator.train()
        self.discriminator.train()

        # Book keeping flags:
        self.num_iterations = num_iterations
        self.num_epochs = int(num_iterations / self.train_dataset.__len__() + 1)

    def __getoptimizer__(self, net):
        return optim.Adam(
            params = filter(lambda p : p.requires_grad, net.parameters()), 
            lr = self.LR
        )

    def adjust_learning_rate(self, factor):
        """Adjust learning rate for later half of the iterations"""
        print("\nDECAYING learning rate.\n")

        # Generator learning rate:
        for param_group in self.gen_optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor

        # Discriminator learning rate:
        for param_group in self.dis_optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor

    @staticmethod
    def content_loss(sr, hr):
        """Compute MSE Content Loss"""
        floss = nn.MSELoss()
        return floss(sr, hr)

    @staticmethod
    def adversarial_loss(predictions, label = 1):
        """Compute Adversarial Binary Cross Entropy Loss"""
        gt = torch.ones_like(predictions) if label is 1 else \
             torch.zeros_like(predictions)
        floss = nn.BCEWithLogitsLoss()
        return floss(predictions, gt)

    def doepoch(self, epoch):
        """
        Step through training
        """
        
        # At around mid-point of the iterations, decay the learning rate:
        if epoch is int(self.num_epochs / 2) + 1:
            self.adjust_learning_rate(factor = 0.1)

        trainbar  = tqdm(self.train_loader)
        gen_loss, disc_loss = [], []
        
        for lrbatch, hrbatch in trainbar:
            # Move to default device:
            lrbatch = lrbatch.to(self.device)
            hrbatch = hrbatch.to(self.device)

            # Forward pass through the generator/discriminator:
            srbatch = self.generator(lrbatch)
            srdisc  = self.discriminator(srbatch)

            # Calculate VGG content loss based on feature maps extracted from 
            # VGG19 for super-resolved (SR) predictions and ground-truth high 
            # resolution (HR) images:
            srfeat = self.vgg(srbatch)
            hrfeat = self.vgg(hrbatch).detach() # Constant = detach
            
            # Calculate total Perceptual loss of the generator:
            perceptual_loss = self.content_loss(srfeat, hrfeat) + \
                              self.loss_factor * self.adversarial_loss(srdisc)

            # Back-propagate the generator and step the optimizer:
            self.gen_optimizer.zero_grad()
            perceptual_loss.backward()
            self.gen_optimizer.step()

            # Now, trigger the discriminator update:
            # NOTE: Here we detach the super-resolution (SR) batch and forward 
            # propagate through the discriminator again. This steps is 
            # performed to avoid accidentally calculating gradients of 
            # generator during the discriminator back-propagation.

            # Discriminate super-resolution (SR) and high-resolution (HR) images
            hrdiscriminated = self.discriminator(hrbatch)
            srdiscriminated = self.discriminator(srbatch.detach())

            # Binary Cross-Entropy discriminator loss:
            discriminator_loss = self.adversarial_loss(srdiscriminated, 0) + \
                                 self.adversarial_loss(hrdiscriminated, 1)

            # Back-propagate discriminator and step optimizer:
            self.dis_optimizer.zero_grad()
            discriminator_loss.backward()
            self.dis_optimizer.step()

            # Book-keeping updates:
            info  = "[TRAIN][Epoch %4d] G: %.4f, D: %.4f"
            info  = info % (epoch, perceptual_loss, discriminator_loss)
            trainbar.set_description(desc = info)
            self.update_board(perceptual_loss, discriminator_loss)

            gen_loss.append(perceptual_loss)
            disc_loss.append(discriminator_loss)

        # Save epoch results:
        self.save(epoch)

        # Evaluate the epoch:
        with torch.no_grad():
            
            # Trackers for SSIM and PSNR:
            ssim, psnr = [], []

            testbar = iter(self.test_loader)
            for lr, hr in testbar:
                # Generate Super-Resolved (SR) image and compute test metrics:
                sr = self.generator(lr)
                ssim.append(pytorch_msssim.ssim(sr, hr, data_range = 1))
                psnr.append(10*log10((hr.max()**2) / self.content_loss(sr,hr)))

                # Print mean PSNR and SSIM results:
                info = '[TEST][Epoch %4d] PSNR: %.4f dB SSIM: %.4f'
                info = info % (mean(psnr), mean(ssim))
                testbar.set_description(desc = info)

        # Return training losses and test metrics for the epoch:
        return mean(gen_loss), mean(disc_loss), mean(ssim), mean(psnr)
    
    def save(self, epoch):
        """Save Generator & Discriminator State"""
        torch.save(self.generator.state_dict(), 'gen_epoch_%d.pth' % epoch)
        torch.save(self.discriminator.state_dict(), 'disc_epoch_%d.pth' % epoch)

    def update_board(self, genloss, disloss):
        # Log losses to tensorboard:
        N = self.num_iterations
        self.logger.add_scalar("LOSS/Generator"    , genloss, N)
        self.logger.add_scalar("LOSS/Discriminator", disloss, N)
