#!/usr/bin/env python3

import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from hdf5_dataset_utils import ConnectivityDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import os
from math import ceil


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_imgs(x, z, y, show=True):
    """
    plot results of model

    inputs:
      x    - the input to the network
      z    - the output of the network
      y    - the desired output
      show - whether or not to call the blocking show function of matplotlib
    """
    assert(x.shape[0] == z.shape[0] == y.shape[0] and z.shape == y.shape)
    layers = z.shape[1]
    rows = x.shape[0]
    cols = 2*layers+1
    for i in range(rows):
        x_tmp = torch.clamp(x[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        z_tmp = torch.clamp(z[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        y_tmp = torch.clamp(y[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        plt.subplot(rows, cols, i*cols+1)
        plt.imshow(x_tmp)
        plt.subplot(rows, cols, i*cols+2)
        plt.imshow(z_tmp)
        plt.subplot(rows, cols, i*cols+3)
        plt.imshow(y_tmp)
    if show:
        plt.show()


class AEBase(pl.LightningModule):
    """base class for connectivity autoencoder models"""

    def __init__(self, log_step):
        super().__init__()

        # cache some data to show learning progress
        self.train_progress_batch = None
        self.val_progress_batch = None

        # want logging frequency less than every training iteration and more
        # than every epoch
        self.log_step = log_step
        self.loss_hist = 0.0
        self.log_it = 0

        # keep track of the best validation models
        self.val_loss_best = float('Inf')
        self.best_model_path = None

    # log network output for a single batch
    def log_network_image(self, batch, name):
        x, y = batch
        y_hat = self.output(x)

        img_list = []
        for i in range(x.shape[0]):
            img_list.append(x[i,...].cpu().detach())
            img_list.append(y_hat[i,...].cpu().detach())
            img_list.append(y[i,...].cpu().detach())

        grid = torch.clamp(make_grid(img_list, nrow=3, padding=20, pad_value=1), 0.0, 1.0)
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def _ckpt_dir(self):
        log_dir = Path(self.trainer.weights_save_path) / self.logger.save_dir
        return log_dir / f'version_{self.logger.version}' / 'checkpoints'

    # provide visual feedback of the learning progress after every epoch
    def training_epoch_end(self, outs):
        torch.set_grad_enabled(False)
        self.eval()

        self.log_network_image(self.train_progress_batch, 'train_results')

        torch.set_grad_enabled(True)
        self.train()

    def validation_epoch_end(self, outs):
        val_loss = np.mean(np.asarray([o.item() for o in outs]))

        # save checkpoint of best performing model
        if val_loss < self.val_loss_best:
            self.val_loss_best = val_loss

            if self.best_model_path is not None:
                self.best_model_path.unlink(missing_ok=True)

            filename = self._ckpt_dir() / f'val_loss_{val_loss:.4f}_epoch_{self.current_epoch}.ckpt'
            self.trainer.save_checkpoint(filename, weights_only=True)
            self.best_model_path = filename

        self.logger.experiment.add_scalar('val_loss', val_loss, self.current_epoch)
        self.log_network_image(self.val_progress_batch, 'val_results')


class UAEModel(AEBase):
    """undercomplete auto encoder for learning connectivity from images"""

    def __init__(self, log_step=1):
        super().__init__(log_step)

        # encoder
        self.econv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.econv2 = nn.Conv2d(4, 8, 5, padding=2)
        self.econv3 = nn.Conv2d(8, 12, 5, padding=2)
        self.econv4 = nn.Conv2d(12, 16, 5, padding=2)
        self.econv5 = nn.Conv2d(16, 20, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        # TODO need more parameters
        # TODO squeeze to vector

        # decoder
        self.dconv1 = nn.Conv2d(20, 16, 5, padding=2)
        self.dconv2 = nn.Conv2d(16, 12, 5, padding=2)
        self.dconv3 = nn.Conv2d(12, 8, 5, padding=2)
        self.dconv4 = nn.Conv2d(8, 4, 5, padding=2)
        self.dconv5 = nn.Conv2d(4, 1, 5, padding=2)

        # TODO randomize initialization

    def forward(self, x):

        # encoder
        x = self.pool(F.relu(self.econv1(x)))
        x = self.pool(F.relu(self.econv2(x)))
        x = self.pool(F.relu(self.econv3(x)))
        x = self.pool(F.relu(self.econv4(x)))
        x = self.pool(F.relu(self.econv5(x)))

        # decoder
        x = F.relu(self.dconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv3(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv4(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv5(F.interpolate(x, scale_factor=2, mode='nearest')))

        return x

    def output(self, x):
        return self(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00001) # NOTE start low and increase
        return optimizer

    def training_step(self, batch, batch_idx):
        # set aside some data to show learning progress
        if self.train_progress_batch is None:
            self.train_progress_batch = batch

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.loss_hist += loss.item()
        if batch_idx != 0 and batch_idx % self.log_step == 0:
            self.logger.experiment.add_scalar('loss', self.loss_hist / self.log_step, self.log_it)
            self.loss_hist = 0.0
            self.log_it += 1

        return loss


class View(nn.Module):
    """helper class to provide view functionality in sequential containers"""
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAEModel(AEBase):
    """Beta Variational Auto Encoder class for learning connectivity from images

    based on BetaVAE_B from https://github.com/1Konny/Beta-VAE/blob/master/model.py
    which is based on: https://arxiv.org/abs/1804.03599

    """

    def __init__(self, beta, z_dim, kld_weight, log_step=1):
        super().__init__(log_step)

        self.beta = beta
        self.kld_weight = kld_weight
        self.z_dim = z_dim # dimension of latent distribution

        self.encoder = nn.Sequential(            #  1, 128, 128 (input)
            nn.Conv2d(1, 32, 4, 2, 1),           # 32,  64,  64
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # 32,  32,  32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # 32,  16,  16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # 32,   8,   8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # 32,   4,   4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # 512
            nn.Linear(32*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 2*z_dim)              # 2 x z_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # 32,   4,   4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # 32,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # 32,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # 32,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # 32,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)   # 32, 128, 128
        )

        # initialize weights
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):

        # encoder spits out latent distribution as a single 32x1 vector with the
        # first 16 elements corresponding to the mean and the last 16 elements
        # corresponding to the log of the variance
        latent_distribution = self.encoder(x)
        mu = latent_distribution[:, :16]
        logvar = latent_distribution[:, 16:]

        # generate the input to the decoder using the reparameterization trick
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        z = mu + std*eps

        out = self.decoder(z)

        return out, mu, logvar

    def output(self, x):
        return self(x)[0]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # set aside some data to show learning progress on training data
        if self.train_progress_batch is None:
            self.train_progress_batch = batch

        x, y = batch
        y_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(y_hat, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + self.beta * self.kld_weight * kld_loss

        self.loss_hist += loss.item()
        if batch_idx != 0 and batch_idx % self.log_step == 0:
            self.logger.experiment.add_scalar('train_loss', self.loss_hist / self.log_step, self.log_it)
            self.loss_hist = 0.0
            self.log_it += 1

        return loss

    def validation_step(self, batch, batch_idx):
        # set aside some data to show network performance on validation data
        if self.val_progress_batch is None:
            self.val_progress_batch = batch

        x, y = batch
        y_hat, _, _ = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss


def train_main(args):

    cpus = os.cpu_count()
    gpus = 1 if torch.cuda.is_available() else 0

    # load dataset

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f'provided dataset {dataset_path} not found')
        return
    train_dataset = ConnectivityDataset(dataset_path, train=True)
    val_dataset = ConnectivityDataset(dataset_path, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=cpus)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=cpus)

    # training params

    beta = 1.0
    z_dim = 16
    kld_weight = 1.0 / len(train_dataloader)
    log_step = ceil(len(train_dataloader)/100) # log averaged loss ~100 times per epoch

    # load model, if provided

    if args.model is not None:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f'provided model {model_path} not found')
            return
        model = BetaVAEModel.load_from_checkpoint(args.model, beta=beta, z_dim=z_dim,
                                                  kld_weight=kld_weight, log_step=log_step)
    else:
        model = BetaVAEModel(beta=beta, z_dim=z_dim, kld_weight=kld_weight, log_step=log_step)

    # train network

    logger = pl_loggers.TensorBoardLogger('runs/', name='')
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, weights_summary='full', gpus=gpus)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='utilities for train and testing a connectivity CNN')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # train subparser
    train_parser = subparsers.add_parser('train', help='train connectivity CNN model on provided dataset')
    train_parser.add_argument('dataset', type=str, help='dataset for training')
    train_parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    train_parser.add_argument('--batch-size', type=int, default=4, help='batch size for training')
    train_parser.add_argument('--model', type=str, help='checkpoint to load')

    args = parser.parse_args()

    if args.command == 'train':
        train_main(args)
