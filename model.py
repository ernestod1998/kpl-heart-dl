from utils import plot_error_map
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import monai
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.metrics import structural_similarity

# define the LightningModule
class UnetModel(pl.LightningModule):
    def __init__(self, input_size, spatial_dims, num_channels, dropout=0.0, lr=1e-3, label_key="kPL"):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.lr = lr
        self.label_key = label_key
        self.model = monai.networks.nets.BasicUNet(spatial_dims=spatial_dims, in_channels=num_channels, out_channels=1, features=(32, 32, 64, 128, 256, 32), dropout=self.dropout) #.to(device)
        #self.model = monai.networks.nets.UNet(spatial_dims=spatial_dims, in_channels=num_channels, out_channels=1, channels=(32, 64, 128, 256), strides=(2,2,2))
        self.testsavepath = "/home/ssahin/kpl-est-dl/test_results/version_20"

        self.l1loss = torch.nn.L1Loss(reduction='none')
        self.mseloss = torch.nn.MSELoss(reduction='none')

    def forward(self, data):
        #ouput of this is [batch_size, (64, 64, 1)]
        #print('forward step')
        #print(data.shape)
        return self.model(data.float())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        imgs = batch["data"] # shape [Nbatch Nt*2 Nx Ny]
        label = batch[self.label_key] # shape [Nbatch Nx Ny]
        #print("training step")
        #print(label)
        prediction = self.forward(imgs).squeeze()
        #print(prediction)

        #loss_l1, loss_sqe = self.calculate_loss(prediction,label)
        loss_l1, loss_sqe = self.calculate_loss(prediction,label)

        # Logging to TensorBoard (if installed) by default
        values = {"train_loss": loss_l1, "train_sq_error": loss_sqe}
        self.log_dict(values, batch_size=2)
        return loss_l1
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        imgs = batch["data"] # shape [N 60 64 64]
        label = batch[self.label_key] # shape [N 64 64]
        #print("validation step")
        #print(label.shape)
        prediction = self.forward(imgs).squeeze()

        #loss_l1, loss_sqe = self.calculate_loss(prediction,label)
        loss_l1, loss_sqe = self.calculate_loss(prediction,label)

        # Logging to TensorBoard (if installed) by default
        values = {"val_loss": loss_l1, "val_sq_error": loss_sqe}
        self.log_dict(values, batch_size=2)


    def test_step(self, batch, batch_idx):
        # this is the validation loop
        imgs = batch["data"] # shape [N 60 64 64]
        label = batch[self.label_key] # shape [N 64 64]
        kTRANS = batch["kTRANS"]
        mask = batch["mask"]
        prediction = self.forward(imgs).squeeze()

        label_np = np.squeeze(label.data.cpu().numpy())
        pred_np = np.squeeze(prediction.data.cpu().numpy())
        imgs_np = np.squeeze(imgs.data.cpu().numpy())
        mask_np = np.squeeze(mask.data.cpu().numpy())
        #kTRANS_np = np.squeeze(kTRANS.data.cpu().numpy())

        l1, sqe, ssim = self.calculate_test_metrics(label_np, pred_np, mask_np)

        #save image
        plot_error_map(label_np, pred_np, imgs_np, mask_np, savepath=os.path.join(self.testsavepath,str(batch_idx+1)+".png"))

        # Logging to TensorBoard (if installed) by default
        values = {"test_loss": l1, "test_sq_error": sqe, "test_ssim": ssim, "SNR_P": batch["SNR"][0][0], "SNR_L": batch["SNR"][0][1]}
        self.log_dict(values, batch_size=1, on_step=True)


    def calculate_loss(self, prediction, label):

        # L1 loss
        output = self.l1loss(prediction, label)
        loss_l1_batch = torch.sum(output, (1,2))
        loss_l1 = torch.mean(loss_l1_batch)

        # squared error loss
        output = self.mseloss(prediction, label)
        loss_sqe_batch = torch.sum(output, (1,2))
        loss_sqe = torch.mean(loss_sqe_batch)

        return loss_l1, loss_sqe
    
    # def calculate_loss_2(self, prediction, label):

    #     # create mask
    #     mask = label
    #     mask[mask>0] =1
    #     mask[mask<=0] =0

    #     pred_masked = prediction*mask
    #     pred_masked = pred_masked[pred_masked!=0]
    #     label_masked = label*mask
    #     label_masked = label_masked[label_masked!=0]

    #     # L1 loss
    #     output = self.l1loss(pred_masked, label_masked) / label_masked
    #     output[output == float('inf')] = 1
    #     loss_l1 = output.sum()

    #     # squared error loss
    #     output = self.mseloss(pred_masked, label_masked) / label_masked
    #     output[output == float('inf')] = 1
    #     loss_sqe = output.sum()

    #     return loss_l1, loss_sqe

    def calculate_test_metrics(self, gt, pred, mask):
        #inputs should be numpy
       
        # create mask
        #mask = ktrans
        #mask[mask>0.1] =1
        #mask[mask<0.1] =0
        mask_nonzero = np.nonzero(mask)
        eps = 1e-12

        #calc abs error
        eps = 1e-12
        abs_error = abs((pred+eps) - (gt+eps)) / (gt+eps)
        l1 = abs_error[mask_nonzero].mean().round(decimals=5)

        #calc sq error
        sq_error = np.square((pred+eps) - (gt+eps)) / (gt+eps)
        sqe = sq_error[mask_nonzero].mean().round(decimals=5)

        #calc ssim
        mssim, S = structural_similarity(pred, gt, data_range=np.max(gt)-np.min(gt), full=True)
        ssim = S[mask_nonzero].mean().round(decimals=5)

        return l1, sqe, ssim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        return optimizer


