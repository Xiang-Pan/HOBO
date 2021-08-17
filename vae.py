'''
Author: Xiang Pan
Date: 2021-08-16 19:26:43
LastEditTime: 2021-08-16 20:56:17
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/vae.py
xiangpan@nyu.edu
'''
from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from task_datasets.siftsmall_datamodule import SIFTSmallDataModule, VecDataset
import os
import numpy as np
# from pytorch_lightning.loggger import wandb
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss


    def test_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    # model = LitAutoEncoder.load_from_checkpoint("./lightning_logs/version_3/checkpoints/epoch=100-step=31612.ckpt")
    model = LitAutoEncoder()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(max_epochs = 100,)
    # print(model.current_epoch)
    # trainer = pl.Trainer(max_epochs=100)
    dm = SIFTSmallDataModule()
    dm.setup()
    trainer.fit(model, dm.train_dataloader())

    dim = 64
    source_path = "./cached_datasets/siftsmall/"
    target_path = "./cached_datasets/siftsmall_"+str(dim)+"/"
    files= os.listdir(source_path)
    os.system("cp "+source_path+"siftsmall_groundtruth.npy"+" "+target_path+"siftsmall_"+str(dim)+"_groundtruth.npy")
    files.remove("siftsmall_groundtruth.npy")
    print(files)
    for f in files:
        src = VecDataset(source_path + f)
        eb_list = []
        for batch in src:
            eb = model(batch)
            eb_list.append(eb)
        new_ebs = torch.stack(eb_list, dim = 0).detach().numpy()
        print(new_ebs.shape)
        dst = (target_path + f.replace("siftsmall", "siftsmall_"+str(dim)))
        np.save(dst, new_ebs)

        


    # trainer.fit(model, dm.train_dataloader())

    # ------------
    # testing
    # ------------
    # result = trainer.validate(model = model, dataloaders=dm.train_dataloader())
    # print(result)


if __name__ == '__main__':
    cli_main()