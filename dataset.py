import torch
import pandas as pd
import h5py
import tqdm
import numpy as np
from torch.utils.data import Dataset

""" Dataset class and read functions"""


class BrainWebDataset(Dataset):

    def __init__(self, group_file, read_fxn):
        """
        Arguments:
            group_file (string): Path to pickle with catalog
            root_dir (string): Directory with all the images.
        """
        self.catalog = pd.read_pickle(group_file)
        self.read_fxn = read_fxn

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.read_fxn(self.catalog.iloc[idx, 0])

        return sample


def catalog_exams(hdf5_files, read_fxn):
    catalog = []
    for h5file in tqdm.tqdm(hdf5_files):
        example_info = read_fxn(h5file, catalog_flag=1)
        catalog.append(example_info)

    catalog = pd.DataFrame(catalog)
    return catalog


def read_basic_sim_data(filepath, idx=None, catalog_flag=0,volumetric_keys=["data", "kPL", "kTRANS"]):
    '''
    '''

    # read in file, initialize arrays
    f = h5py.File(filepath, 'r')
    exam_dict = {"file": filepath, "kPL_lim": f.attrs["kPL_lim"], "kTRANS_lim": f.attrs["kTRANS_lim"], "std_noise": f.attrs["std_noise"]}

    if not catalog_flag:
        exam_dict["data"] = np.transpose(np.reshape(f["data"][:], (64, 64, 60)), (2, 0, 1))
        exam_dict["kPL"] = np.squeeze(f["kPL"][:])
        exam_dict["kTRANS"] = f["kTRANS"][:]

    f.close()

    return exam_dict


def read_brainweb_sim_data(filepath, idx=None, catalog_flag=0, mask_flag=0, volumetric_keys=["metImages", "kMaps", "kTRANS"]):
    '''
    '''

    # read in file, initialize arrays
    f = h5py.File(filepath, 'r')
    exam_dict = {"file": filepath, "kineticRates": f.attrs["kineticRates"], "ktransScales": f.attrs["ktransScales"], "SNR": f.attrs["SNR"], "Tarrival": f.attrs["Tarrival"], "coil_lim": f.attrs["coil_lim"], "Mz0_scale": f.attrs["Mz0_scale"], "brain_idx": f.attrs["brain_idx"]}

    if not catalog_flag:
        met_img = np.reshape(np.transpose(f["metImages"][:,:,:,0:2], (0, 1, 3, 2)), (64, 64, 40))
        exam_dict["data"] = np.transpose(met_img, (2, 0, 1))
        exam_dict["kPL"] = np.squeeze(f["kMaps"][0,:,:])
        exam_dict["kTRANS"] = f["kTRANS"][:]
        if mask_flag:
            exam_dict["mask"] = f["mask"][:]
    
    f.close()

    return exam_dict


def read_sim_data_comp(filepath, idx=None, catalog_flag=0,volumetric_keys=["data", "kPL", "kTRANS"]):
    '''
    '''

    # read in file, initialize arrays
    f = h5py.File(filepath, 'r')
    exam_dict = {"file": filepath, "kPL_lim": f.attrs["kPL_lim"], "kTRANS_lim": f.attrs["kTRANS_lim"], "std_noise": f.attrs["std_noise"]}

    exam_dict["data"] = np.transpose(np.reshape(f["data"][:], (64, 64, 60)), (2, 0, 1))
    exam_dict["kPL"] = np.squeeze(f["kPL"][:])
    exam_dict["kTRANS"] = f["kTRANS"][:]
    exam_dict["kPL_PK"] = f["kPL_PK"][:]
    exam_dict["kPL_denoise_PK"] = f["kPL_denoise_PK"][:]

    f.close()

    return exam_dict


def read_brainweb_sim_data_comp(filepath, idx=None, catalog_flag=0,volumetric_keys=["data", "kPL", "kTRANS"]):
    '''
    '''

    # read in file, initialize arrays
    f = h5py.File(filepath, 'r')
    exam_dict = {"file": filepath, "kineticRates": f.attrs["kineticRates"], "ktransScales": f.attrs["ktransScales"], "SNR": f.attrs["SNR"], "Tarrival": f.attrs["Tarrival"], "coil_lim": f.attrs["coil_lim"], "Mz0_scale": f.attrs["Mz0_scale"], "brain_idx": f.attrs["brain_idx"]}

    met_img = np.reshape(np.transpose(f["metImages"][:,:,:,0:2], (0, 1, 3, 2)), (64, 64, 40))
    exam_dict["data"] = np.transpose(met_img, (2, 0, 1))
    exam_dict["kPL"] = np.squeeze(f["kMaps"][0,:,:])
    exam_dict["kTRANS"] = f["kTRANS"][:]
    exam_dict["kPL_PK"] = f["kPL_PK"][:]
    exam_dict["kPL_denoise_PK"] = f["kPL_denoise_PK"][:]
    exam_dict["mask"] = f["mask"][:]
        
    f.close()

    return exam_dict


def read_data_invivo(filepath, idx=None, catalog_flag=0,volumetric_keys=["data", "kTRANS"]):
    '''
    '''

    # read in file, initialize arrays
    f = h5py.File(filepath, 'r')
    exam_dict = {"file": filepath, "std_noise_pyr": f.attrs["std_noise_pyr"], "std_noise_lac": f.attrs["std_noise_lac"], "std_noise_bic": f.attrs["std_noise_bic"]}

    met_img = np.reshape(np.transpose(f["metImages"][:,:,:,0:2], (0, 1, 3, 2)), (64, 64, 40))
    exam_dict["data"] = np.transpose(met_img, (2, 0, 1))

    exam_dict["kPL_PK"] = f["kPL_PK"][:]
    exam_dict["lac_rsq_PK"] = f["lac_rsq_PK"][:]
    exam_dict["kPL_denoise_PK"] = f["kPL_denoise_PK"][:]
    exam_dict["kPL_const"] = f["kPL_const"][:]
    exam_dict["mask"] = f["mask"][:]

    f.close()

    return exam_dict


# def read_brainweb_sim_data_resize(filepath, idx=None, catalog_flag=0,volumetric_keys=["metImages", "kMaps", "kTRANS"]):
#     '''
#     '''

#     # read in file, initialize arrays
#     f = h5py.File(filepath, 'r')
#     exam_dict = {"file": filepath, "kineticRates": f.attrs["kineticRates"], "ktransScales": f.attrs["ktransScales"], "std_noise": f.attrs["std_noise"]}

#     if not catalog_flag:
#         met_img = np.reshape(np.transpose(f["metImages"][:,:,:,0:2], (0, 1, 3, 2)), (16, 16, 40))
#         exam_dict["data"] = np.transpose(met_img, (2, 0, 1))
#         exam_dict["kPL"] = np.squeeze(f["kMaps"][0,:,:])
#         exam_dict["kTRANS"] = f["kTRANS"][:]

#         exam_dict["kTRANS"] = transform.resize(exam_dict["kTRANS"], (64,64))
#         exam_dict["kPL"] = transform.resize(exam_dict["kPL"], (64,64))
#         exam_dict["data"] = transform.resize(exam_dict["data"], (40,64,64))
    
#     f.close()

#     return exam_dict