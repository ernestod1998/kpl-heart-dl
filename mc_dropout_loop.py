from utils import  monte_carlo_dropout_analysis, plot_comp_mc_dropout
from dataset import read_data_invivo
import glob
import matplotlib.pyplot as plt
from model import UnetModel

plt.rcParams.update({'font.size': 20})

# init the classification model
input_size = (64, 64)
spatial_dims = 2 
num_channels = 40 #metabolies*tpts

model = UnetModel.load_from_checkpoint("/home/ssahin/kpl-est-dl/checkpoints/version_21/epoch=2999-val_loss=3.66.ckpt", input_size=input_size, spatial_dims=spatial_dims, num_channels=num_channels, dropout=0.3)

model.eval()
for m in model.modules():  #set Dropout layers to train to allow dropout in eval mode
    if m.__class__.__name__.startswith('Dropout'):
        m.train()

data_path_test = "/data/ssahin/kpl_dl_sim/brainweb_9_2/invivo_test/"
list_val = glob.glob(data_path_test + '/*.h5')

savepath = "/home/ssahin/kpl-est-dl/test_results/version_21/mc_dropout_2/"
for path in list_val:
    dict_val = read_data_invivo(path)
    num = path.split("/")[-1].split(".")[0]
    [mean_map, var_map, kpl_preds] = monte_carlo_dropout_analysis(dict_val["data"], model, its=1000, map_size=[64, 64])
    plot_comp_mc_dropout(dict_val["data"], mean_map, var_map, savepath=(savepath+num+".png"))
