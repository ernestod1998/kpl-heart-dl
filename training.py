from dataset import BrainWebDataset, read_brainweb_sim_data
import os
import pytorch_lightning as pl
from model import UnetModel
from torch.utils.data import DataLoader

# init the classification model
input_size = (64, 64)
spatial_dims = 2 
num_channels = 40 #metabolies*tpts
unet_model = UnetModel(input_size=input_size, spatial_dims=spatial_dims, num_channels=num_channels, dropout=0.3)
logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./checkpoints/version_20", filename='{epoch}-{val_loss:.2f}', save_top_k=3, monitor="val_loss") #change naming of checkpt to include val loss

# dataloader
train_dataset = BrainWebDataset("/home/ssahin/kpl-est-dl/groups/train_BW_9_2.pkl", read_brainweb_sim_data)
val_dataset = BrainWebDataset("/home/ssahin/kpl-est-dl/groups/val_BW_9_2.pkl", read_brainweb_sim_data)

dl_train = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
dl_val = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)

# start training
trainer = pl.Trainer(max_epochs=5, devices=[0], accelerator="gpu", logger=logger, callbacks=[checkpoint_callback]) #can train on gpu set here, setting devices=2 didnt work #[1] = 0 on nvtop(works best) and [0]=1 on nvtop
trainer.fit(model=unet_model, train_dataloaders=dl_train, val_dataloaders=dl_val) 