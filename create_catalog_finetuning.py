import glob
from dataset import read_data_invivo, catalog_exams
from sklearn.model_selection import train_test_split
import os

save_path = '/home/ssahin/kpl-est-dl/groups'

hdf5_dir = "/data/ssahin/kpl_dl_sim/brainweb_9_2/invivo_train"
h5files = glob.glob(hdf5_dir + '/*.h5')

catalog = catalog_exams(h5files, read_data_invivo)

train, val = train_test_split(catalog, test_size=13, random_state=82)

train.to_pickle(os.path.join(save_path,"train_invivo_9_2.pkl"))
val.to_pickle(os.path.join(save_path,"val_invivo_9_2.pkl"))
