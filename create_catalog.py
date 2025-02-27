import glob
from dataset import read_brainweb_sim_data, catalog_exams
from sklearn.model_selection import train_test_split
import os

save_path = '/home/ssahin/kpl-est-dl/groups'

hdf5_dir = "/data/ssahin/kpl_dl_sim/brainweb_9_2/sim_data_trainval"
h5files = glob.glob(hdf5_dir + '/*.h5')

catalog = catalog_exams(h5files, read_brainweb_sim_data)

#trainval, test = train_test_split(catalog, test_size=400, random_state=888)
train, val = train_test_split(catalog, test_size=400, random_state=888)

train.to_pickle(os.path.join(save_path,"train_BW_9_2.pkl"))
val.to_pickle(os.path.join(save_path,"val_BW_9_2.pkl"))


hdf5_dir_test = "/data/ssahin/kpl_dl_sim/brainweb_9_2/sim_data_test"
h5files_test = glob.glob(hdf5_dir_test + '/*.h5')

test = catalog_exams(h5files_test, read_brainweb_sim_data)
test.to_pickle(os.path.join(save_path,"test_BW_9_2.pkl"))
