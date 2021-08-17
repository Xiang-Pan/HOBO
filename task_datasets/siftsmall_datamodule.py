'''
Author: Xiang Pan
Date: 2021-08-16 19:32:16
LastEditTime: 2021-08-16 20:38:33
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/task_datasets/siftsmall_datamodule.py
xiangpan@nyu.edu
'''
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
class VecDataset(Dataset):
    def __init__(self, path = None, train = True):
        if path is not None:

            self.data = torch.Tensor(np.load(path))
        else:
            self.data = torch.Tensor(np.load("./cached_datasets/siftsmall/siftsmall_base.npy"))
        # self.labels = labels

    def __getitem__(self, index):
        return self.data[index]
        # return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
        
class SIFTSmallDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./cached_datasets/siftsmall/siftsmall_base.npy", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage = None):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        self.train_dataset = VecDataset(train=True)
        # self.tes
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        # ...

if __name__ == '__main__':
    datamodule = SIFTSmallDataModule()
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(batch.shape)
    # dataset = VecDataset(train=True)
    