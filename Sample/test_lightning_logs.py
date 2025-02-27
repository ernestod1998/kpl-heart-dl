import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


#1)Define a Radom Dataset
class RandomDataset(Dataset):
    def __init__(self, num_sample = 100,input_dim = 10):
        super().__init__()
        self.num_samples = num_sample
        self.input_dim = input_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randn(1)
        return x,y
 #2) Create Lightning Model
class MyLightningModel(pl.LightningModule):
        def __init__(self, input_dim=10,hidden_dim=16):
            super().__init__()

            self.net = nn.Sequential(
              nn.Linear(input_dim,hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim,1)  
            )
            self.loss_fn = nn.MSELoss()
        
        def forward(self,x):
            return self.net(x)
        
        def training_step(self,batch,batch_idx):
         x,y = batch
         y_hat = self(x)
         train_loss = self.loss_fn(y_hat,y)

         self.log('train_loss', train_loss, on_step = True,on_epoch = True,prog_bar = True)
         return train_loss    

        def validation_step(self, batch, batch_idx):
         x,y = batch
         y_hat = self(x)
         val_loss = self.loss_fn(y_hat, y)

         self.log('val_loss', val_loss, on_step = False, on_epoch=True, prog_bar=True)
         return val_loss

        def test_step(self,batch,batch_idx):
            x,y = batch
            y_hat = self(x)
            test_loss = self.loss_fn(y_hat,y)

            self.log('test_loss', test_loss, on_epoch=True,prog_bar=True)
            return test_loss

        def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(),lr=1e-3,)  

  #3 Create data loaders,model,trainer, and run
if __name__ == "__main__":
    print("hi")
    train_dataset = RandomDataset(num_sample=500,input_dim=10)
    val_dataset = RandomDataset(num_sample=100, input_dim=10)
    test_dataset = RandomDataset(num_sample=50,input_dim=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    for batch in train_loader:
     x, y = batch  # x.shape = [32, 10], y.shape = [32, 1]
     print(x, y)
     break

    tb_logger = TensorBoardLogger("logs", name = "my_experiment")

    model = MyLightningModel(input_dim = 10, hidden_dim = 16)

    #Create a trainer with 3 epochs
    print("hi2")
    trainer = pl.Trainer(
        max_epochs = 3,
        accelerator = "gpu",
        devices = 1,
        logger = tb_logger,
        enable_progress_bar = True
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("starting test...")
    trainer.test(model, dataloaders=test_loader)

