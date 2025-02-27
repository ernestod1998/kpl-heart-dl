from dataset import BrainWebDataset, read_data_invivo
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from model import UnetModel
from torch.utils.data import DataLoader, Dataset

# init the classification model
input_size = (64, 64)
spatial_dims = 2 
num_channels = 40 #metabolies*tpts
unet_model = UnetModel.load_from_checkpoint("/home/ssahin/kpl-est-dl/checkpoints/version_20/epoch=213-val_loss=2.23.ckpt", input_size=input_size, spatial_dims=spatial_dims, num_channels=num_channels, dropout=0.3, lr=1e-5, label_key="kPL_PK")
#unet_model = UnetModel.load_from_checkpoint("/home/ssahin/kpl-est-dl/checkpoints/version_17/epoch=2995-val_loss=1.50.ckpt", input_size=input_size, spatial_dims=spatial_dims, num_channels=num_channels, dropout=0.3, lr=1e-5, label_key="kPL_PK")
logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./checkpoints/version_21", filename='{epoch}-{val_loss:.2f}', save_top_k=10, monitor="val_loss", every_n_epochs=300) 

# dataloader
train_dataset = BrainWebDataset("groups/train_invivo_9_2.pkl", read_data_invivo)
val_dataset = BrainWebDataset("groups/val_invivo_9_2.pkl", read_data_invivo)

dl_train = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
dl_val = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)

# start training
trainer = pl.Trainer(max_epochs=3000, devices=[1], accelerator="gpu", logger=logger, callbacks=[checkpoint_callback]) #can train on gpu set here, setting devices=2 didnt work #[1] = 0 on nvtop(works best) and [0]=1 on nvtop
trainer.fit(model=unet_model, train_dataloaders=dl_train, val_dataloaders=dl_val)