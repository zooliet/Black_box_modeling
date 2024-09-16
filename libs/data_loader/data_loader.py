
import os
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split

# import torchvision.transforms as transforms

class BBDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        if 'ID' in self.df.columns:
            self.df.pop('ID')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print('idx:', idx)
        row = self.df.iloc[idx]
        row = torch.tensor(row.values, dtype=torch.float32)
        
        X = row[:-1] 
        y = row[-1] 
        return X, y


class BBDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=32, num_workers=4):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # if stage == 'fit' or stage is None:
        #     pass
        # if stage == 'test' or stage is None:
        #     pass

        self.train_ds = BBDataset(self.csv_file)
        self.train_ds, self.val_ds = random_split(self.train_ds, [0.8, 0.2])
        # self.train_ds, self.val_ds = random_split(self.train_ds, [1000, 35118])
        self.test_ds = self.val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

