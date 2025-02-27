import os
import h5py
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import torch


def plot_error_map(kpl_gt, kpl_pred, data, mask=None, **kwargs):

    savepath = kwargs.get('savepath', None)
    clims = kwargs.get('clims', [0, np.max(kpl_gt)])
    ktrans = kwargs.get('ktrans', None)

    if mask is None:
        # create mask
        mask = ktrans
        mask[mask>0.05] =1

    eps = 1e-12
    error_map = abs(((kpl_gt+eps) - (kpl_pred+eps)) / (kpl_gt+eps))
    error_map_masked = error_map * mask
    #print(np.max(error_map_masked))
    #print(np.min(error_map_masked))

    #calc abs error
    #mask_nonzero = np.nonzero(mask)
    mean_error = error_map_masked[np.nonzero(mask)].mean().round(decimals=3)

    plt.rcParams.update({'font.size': 16})

    #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(35,5), ncols=5, nrows=1)  
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(figsize=(35,5), ncols=6, nrows=1)  

    pyr = np.squeeze(np.sum(data[0:19, :, :],axis=0))
    #pyr = np.squeeze(data[2, :, :])
    img1 = ax1.imshow(pyr, cmap="inferno", vmax=np.percentile(pyr, 99.6))
    ax1.set_title('Pyruvate AUC')
    fig.colorbar(img1, ax=ax1)
    ax1.axis('off')

    lac = np.squeeze(np.sum(data[19:40, :, :],axis=0))
    #lac = np.squeeze(data[24, :, :])
    img2 = ax2.imshow(lac, cmap="inferno")
    ax2.set_title('Lactate AUC')
    fig.colorbar(img2, ax=ax2)
    ax2.axis('off')

    img3 = ax3.imshow(kpl_gt, cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax3.set_title('Ground Truth kPL Map')
    fig.colorbar(img3, ax=ax3)
    ax3.axis('off')

    img4 = ax4.imshow(kpl_pred, cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax4.set_title('Predicted kPL Map')
    fig.colorbar(img4, ax=ax4)
    ax4.axis('off')

    img5 = ax5.imshow(error_map_masked, cmap="Greys_r", vmin=0, vmax=np.percentile(error_map_masked,95))
    ax5.set_title('Norm Abs Error Map, \n Mean Abs Error= '+str(mean_error))
    fig.colorbar(img5, ax=ax5)
    ax5.axis('off')

    img6 = ax6.imshow(mask, cmap="Greys_r", vmin=0, vmax=1.0)
    ax6.set_title('Mask')
    fig.colorbar(img6, ax=ax6)
    ax6.axis('off')

    plt.show()

    if savepath:
        fig.savefig(savepath)


def plot_comp(data, kpl_gt, kpl_pred, kpl_pk, kpl_pk_dn, mask, **kwargs):

    plt.rcParams.update({'font.size': 12})
    savepath = kwargs.get('savepath', None)
    clims = kwargs.get('clims', [0, np.max(kpl_gt)])
    mask_thresh= kwargs.get('mask_thresh', 0)

    # if mask is None:
    #     # create mask
    #     mask = ktrans
    #     mask[mask>mask_thresh] =1
    #     mask[mask<=mask_thresh] =0

    mask_nonzero = np.nonzero(mask)
    eps = 1e-12

    error_map_pred = (((kpl_pred+eps) - (kpl_gt+eps)) / (kpl_gt+eps)) * mask
    mean_error_pred = np.abs(error_map_pred[mask_nonzero]).mean().round(decimals=3)

    error_map_pk = (((kpl_pk+eps) - (kpl_gt+eps)) / (kpl_gt+eps)) * mask
    mean_error_pk = np.abs(error_map_pk[mask_nonzero]).mean().round(decimals=3)

    error_map_pkdn = (((kpl_pk_dn+eps) - (kpl_gt+eps)) / (kpl_gt+eps)) * mask
    mean_error_pkdn = np.abs(error_map_pkdn[mask_nonzero]).mean().round(decimals=3)

    limits_err = np.abs([np.percentile(error_map_pred, 1), np.percentile(error_map_pred, 99)])
    lim_err = np.max(limits_err)

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(figsize=(15,5), ncols=5, nrows=2)  

    pyr = np.rot90(np.squeeze(np.sum(data[0:19, :, :],axis=0)), k=-1)
    img1 = ax1.imshow(pyr, cmap="inferno", vmax=np.percentile(pyr, 99.6))
    ax1.set_title('Pyruvate AUC')
    fig.colorbar(img1, ax=ax1)
    ax1.axis('off')

    img2 = ax2.imshow(np.rot90(kpl_gt, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax2.set_title('Ground Truth kPL Map')
    fig.colorbar(img2, ax=ax2)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img3 = ax3.imshow(np.rot90(kpl_pred, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax3.set_title('UNet Predicted kPL Map')
    fig.colorbar(img3, ax=ax3)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img4 = ax4.imshow(np.rot90(kpl_pk*mask, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax4.set_title('Voxelwise PK Model kPL Map')
    fig.colorbar(img4, ax=ax4)
    ax4.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img5 = ax5.imshow(np.rot90(kpl_pk_dn*mask, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax5.set_title('Voxelwise PK Model + \n HOSVD denoising kPL Map')
    fig.colorbar(img5, ax=ax5)
    ax5.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    lac = np.rot90(np.squeeze(np.sum(data[19:40, :, :],axis=0)), k=-1)
    img6 = ax6.imshow(lac, cmap="inferno")
    ax6.set_title('Lactate AUC')
    fig.colorbar(img6, ax=ax6)
    ax6.axis('off')

    ax7.axis('off')

    img8 = ax8.imshow(np.rot90(error_map_pred, k=-1), cmap="bwr", vmin=-1*lim_err, vmax=lim_err)
    ax8.set_title('Error Map, \n Mean Abs Err= '+str(mean_error_pred))
    fig.colorbar(img8, ax=ax8)
    ax8.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)
    

    img9 = ax9.imshow(np.rot90(error_map_pk, k=-1), cmap="bwr", vmin=-1*lim_err, vmax=lim_err)
    ax9.set_title('Error Map, \n Mean Abs Err= '+str(mean_error_pk))
    fig.colorbar(img9, ax=ax9)
    ax9.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img10 = ax10.imshow(np.rot90(error_map_pkdn, k=-1), cmap="bwr", vmin=-1*lim_err, vmax=lim_err)
    ax10.set_title('Abs. Error Map, \n Mean Abs Err= '+str(mean_error_pkdn))
    fig.colorbar(img10, ax=ax10)
    ax10.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath)


def plot_comp_invivo(data, kpl_pred, kpl_const, kpl_pk, kpl_pk_dn, mask, **kwargs):

    savepath = kwargs.get('savepath', None)
    clims = kwargs.get('clims', [0, np.max(kpl_pk)])


    fig, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(figsize=(30,4), ncols=6, nrows=1)  

    pyr = np.rot90(np.squeeze(np.sum(data[:20, :, :],axis=0)), k=-1)
    img1 = ax1.imshow(pyr, cmap="inferno", vmax=np.percentile(pyr, 99.6))
    ax1.set_title('Pyruvate AUC')
    fig.colorbar(img1, ax=ax1)
    ax1.axis('off')

    lac = np.rot90(np.squeeze(np.sum(data[20:, :, :],axis=0)), k=-1)
    img2 = ax2.imshow(lac, cmap="inferno")
    ax2.set_title('Lactate AUC')
    fig.colorbar(img2, ax=ax2)
    ax2.axis('off')

    img3 = ax3.imshow(np.rot90(kpl_pred, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax3.set_title('UNet Predicted kPL Map')
    fig.colorbar(img3, ax=ax3)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img4 = ax4.imshow(np.rot90(kpl_const, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax4.set_title('Spatio-Temporally Constrained \n kPL Map')
    fig.colorbar(img4, ax=ax4)
    ax4.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img5 = ax5.imshow(np.rot90(kpl_pk*mask, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax5.set_title('Voxelwise PK Model kPL Map')
    fig.colorbar(img5, ax=ax5)
    ax5.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img6 = ax6.imshow(np.rot90(kpl_pk_dn*mask, k=-1), cmap="inferno", vmin=clims[0], vmax=clims[1])
    ax6.set_title('Voxelwise PK Model + HOSVD \n denoising kPL Map')
    fig.colorbar(img6, ax=ax6)
    ax6.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath)


def plot_comp_invivo_multi(data, kpl_preds, kpl_const, kpl_pk, kpl_pk_dn, mask, **kwargs):

    epoch_num = kwargs.get('epoch_num', [300, 600, 900])
    savepath = kwargs.get('savepath', None)
    clims = kwargs.get('clims', [0, np.max(kpl_pk)])
    cmap='inferno'

    num_preds = len(kpl_preds)

    fig, axs = plt.subplots(figsize=(4*(num_preds+6),4), ncols=num_preds+5, nrows=1)  

    pyr = np.rot90(np.squeeze(np.sum(data[:20, :, :],axis=0)), k=-1)
    img1 = axs[0].imshow(pyr, cmap=cmap, vmax=np.percentile(pyr, 99.6))
    axs[0].set_title('Pyruvate AUC')
    fig.colorbar(img1, ax=axs[0])
    axs[0].axis('off')

    lac = np.rot90(np.squeeze(np.sum(data[20:, :, :],axis=0)), k=-1)
    img2 = axs[1].imshow(lac, cmap=cmap)
    axs[1].set_title('Lactate AUC')
    fig.colorbar(img2, ax=axs[1])
    axs[1].axis('off')

    for i,kpl_pred in enumerate(kpl_preds):
        img = axs[2+i].imshow(np.rot90(kpl_pred, k=-1), cmap=cmap, vmin=clims[0], vmax=clims[1])
        axs[2+i].set_title('UNet Predicted kPL Map\n Epoch #'+str(epoch_num[i]))
        fig.colorbar(img, ax=axs[2+i])
        axs[2+i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img = axs[3+i].imshow(np.rot90(kpl_const, k=-1), cmap=cmap, vmin=clims[0], vmax=clims[1])
    axs[3+i].set_title('Spatio-Temporally \n Constrained kPL Map')
    fig.colorbar(img, ax=axs[3+i])
    axs[3+i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img = axs[4+i].imshow(np.rot90(kpl_pk*mask, k=-1), cmap=cmap, vmin=clims[0], vmax=clims[1])
    axs[4+i].set_title('Voxelwise PK Model kPL Map')
    fig.colorbar(img, ax=axs[4+i])
    axs[4+i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    img = axs[5+i].imshow(np.rot90(kpl_pk_dn*mask, k=-1), cmap=cmap, vmin=clims[0], vmax=clims[1])
    axs[5+i].set_title('Voxelwise PK Model + HOSVD \n denoising kPL Map')
    fig.colorbar(img, ax=axs[5+i])
    axs[5+i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft = False, labelright=False)

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath)


def resample_brainweb_sim_data(filepath):
    '''
    '''

    # read in file, initialize arrays
    f = h5py.File(filepath, 'r')

    data = transform.resize(f["metImages"][:], (64,64,20,3))
    kPL = transform.resize(f["kMaps"][:], (2,64,64))
    kTRANS = transform.resize(f["kTRANS"][:], (64,64))
    
    f.close()

    f = h5py.File(filepath, 'a')

    f.create_dataset('metImages_64', data=data)
    f.create_dataset('kPL_64', data=kPL)
    f.create_dataset('kTRANS_64', data=kTRANS)

    f.close()


def multiframeimshow(images, scale, dims, customCMAP=None, mask=None):
    """
    This module was written to store potentially useful visualization tools for
    hp data (aka high dimensional image data)
    author: anna bennett
    date: 05.30.2024
    """
    fig, axs = plt.subplots(
        dims[0], dims[1], facecolor="w", edgecolor="k", tight_layout=True
    )

    if customCMAP is None:
        colors = "viridis"
    else:
        colors = customCMAP

    if mask is not None:
        images[mask == 0] = np.nan

    for ax, i in zip(axs.ravel(), range(images.shape[2])):
        ax.set_xlim(0, images.shape[0] - 1)
        ax.set_ylim(0, images.shape[1] - 1)
        ax.imshow(
            images[:, :, i],
            vmin=scale[0],
            vmax=scale[1],
            origin="upper",
            cmap=colors,
            extent=[0, images.shape[0], 0, images.shape[1]],
            aspect=1,
        )

        y_ax = ax.axes.get_yaxis()
        y_ax.set_visible(False)
        x_ax = ax.axes.get_xaxis()
        x_ax.set_visible(False)

    return fig


def monte_carlo_dropout_analysis(data, model, its, map_size=[64, 64]):
    
    x = torch.from_numpy(data)
    x = x[None, :, :, :]

    preds = np.zeros(map_size + [its])
    for i in tqdm.tqdm(range(its)):
        with torch.no_grad():
            preds[:,:,i] = np.squeeze(model(x).numpy())

    mean_map = np.mean(preds, axis=2)
    variance_map = np.var(preds, axis=2)

    return mean_map, variance_map, preds


def plot_comp_mc_dropout(data, mean_map, var_map, **kwargs):

    savepath = kwargs.get('savepath', None)
    clims = kwargs.get('clims', [0, np.max(mean_map)])
    cmap = "inferno"
    

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(figsize=(18,4), ncols=4, nrows=1)  

    pyr = np.rot90(np.squeeze(np.sum(data[:20, :, :],axis=0)), k=-1)
    lac = np.rot90(np.squeeze(np.sum(data[20:, :, :],axis=0)), k=-1)

    img1 = ax1.imshow(pyr, cmap=cmap, vmax=np.percentile(pyr, 99.6))
    ax1.set_title('Pyruvate AUC')
    fig.colorbar(img1, ax=ax1)
    ax1.axis('off')

    img2 = ax2.imshow(lac, cmap=cmap)
    ax2.set_title('Lactate AUC')
    fig.colorbar(img2, ax=ax2)
    ax2.axis('off')

    img3 = ax3.imshow(np.rot90(mean_map, k=-1), cmap=cmap, vmin=clims[0], vmax=clims[1])
    ax3.set_title('MC Dropout \n Mean Map   ')
    fig.colorbar(img3, ax=ax3)
    ax3.axis('off')

    var_map = var_map / pyr
    img4 = ax4.imshow(np.rot90(var_map, k=-1), vmin=0, vmax=np.percentile(var_map, 99), cmap=cmap)
    ax4.set_title('MC Dropout Normalized \n Variance Map       ')
    fig.colorbar(img4, ax=ax4)
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath)