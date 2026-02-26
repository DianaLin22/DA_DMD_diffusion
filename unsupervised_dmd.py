####### final code pixel by pixel unsupervised dmd

import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from utils import standardize_minmax, normalize
from io import StringIO
import pandas as pd
import tifffile
from skimage import exposure, segmentation, filters
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from pydmd import MrDMD, DMD, OptDMD
from pydmd.plotter import plot_eigs_mrdmd
from scipy.signal import correlate
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter



def get_hankel_matrix(data, Ndelay):
    n = len(data)
    H = np.zeros((Ndelay, n - Ndelay + 1))
    for i in range(Ndelay):
        H[i, :] = data[i:i + n - Ndelay + 1]
    return H


def autocorrelation_spectrum(spectrum, max_lag=100):
    acf = correlate(spectrum - np.mean(spectrum), spectrum - np.mean(spectrum), mode='full')
    acf = acf[len(acf)//2:len(acf)//2 + max_lag]  # Keep only positive lags
    return acf / acf[0]  # Normalize


def ramdmd(spectrum, Ndelay, mode_indices=[2], pad_flip=True):
    pad = Ndelay - 1
    cars = normalize(spectrum)
    if pad_flip == True:
        cars = np.pad(np.flip(cars), (pad, 0), mode='constant')
    Hx = get_hankel_matrix(cars, Ndelay).T
    # Apply DMD
    dmd = DMD(svd_rank=Ndelay)
    dmd.fit(Hx)

    modes1 = dmd.modes[:, mode_indices]
    dynamics1 = dmd.dynamics[mode_indices, :]

    dmd_reconstructed = np.abs(np.sum(modes1 @ dynamics1, axis=1))

    if pad_flip == True:
        return np.flip(normalize(dmd_reconstructed))
    else:
        return normalize(dmd_reconstructed)




if __name__ == "__main__":

    Ndelay = 40
    n_clusters = 2
    fs = 30

    cars = standardize_minmax(np.load('synthetic_data/1_synthetic_cars.npy'))[0, :]
    raman = standardize_minmax(np.load('synthetic_data/1_synthetic_raman.npy'))[0, :]


    spectrum = np.pad(cars, (Ndelay-1, 0), mode='constant')
    # acf = autocorrelation_spectrum(spectrum)


    ################# Plot Results
    # fig = plt.subplots(1, 1, figsize=(15, 4))
    # plt.plot(acf, color="#4D4D4D", marker='o', linewidth=2)
    # plt.axvline(x = 12, color = 'r', linestyle = '--')
    # plt.xlabel('Lag')
    # plt.ylabel('ACF')
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.tick_params(axis='x')
    # plt.tick_params(axis='y')
    # plt.tight_layout()
    # plt.show()

    pad = Ndelay-1

    Hx = get_hankel_matrix(cars, Ndelay).T

    # Apply DMD
    dmd = DMD(svd_rank=Ndelay)
    dmd.fit(Hx)

    omega = np.imag(dmd.eigs)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(omega.reshape(-1, 1))
    om_class = km.labels_


    fig, ax = plt.subplots(nrows=8, ncols=1, sharex=True, figsize=(25, 20))
    ax[0].plot(cars, color="#d62728", label='CARS')
    ax[0].legend()
    ax[1].plot(raman, color='g', label='Raman')
    ax[1].legend()
    for i in range(6):
        ax[2+i].plot(5 * np.abs(dmd.modes[:, i]), color="#0072B2", label=f'Mode {i+1}')
        ax[2+i].legend()

    ax[3].set_ylabel("Intensity (a.u.)", labelpad=50)
    ax[7].set_xlabel("Wavenumber ($cm^{-1}$)", labelpad=40)
    plt.show()


    reconstructed_clusters = {i: np.zeros_like(Hx) for i in range(n_clusters)}

    for cluster in range(n_clusters):
        cluster_indices = np.where(om_class == cluster)[0]
        modes1 = dmd.modes[:, cluster_indices]
        dynamics1 = dmd.dynamics[cluster_indices, :]

        reconstructed_clusters[cluster] = 5 * np.abs(np.sum(modes1 @ dynamics1, axis=1))


    mod_indices = np.arange(2, 3, 1)
    modes1 = dmd.modes[:, mod_indices]
    dynamics1 = dmd.dynamics[mod_indices, :]

    dmd_reconstructed = np.abs(np.sum(modes1 @ dynamics1, axis=1))


    ##Plot reconstruction
    plt.figure(figsize=(13, 5))
    for cluster in range(n_clusters):
        res = normalize(reconstructed_clusters[cluster][:])
        if cluster == 1:
            plt.plot( res, label=f'Low Frequency Cluster', linewidth=0.8, color='r')
        if cluster == 0:
            plt.plot(res, label=f'High Frequency Cluster', linewidth=0.8, color='b')


    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.legend(frameon=False)
    plt.show()


    ind = om_class[0]
    cluster_indices = np.where(om_class != ind)[0]
    modes1 = dmd.modes[:, cluster_indices]
    dynamics1 = dmd.dynamics[cluster_indices, :]
    reconstructed_clusters = 5 * np.abs(np.sum(modes1 @ dynamics1, axis=1))


    shift = Ndelay // 2
    data1 = np.roll(reconstructed_clusters[:], 0)


    fig, ax = plt.subplots(nrows=2 , ncols=1, sharex=True, figsize=(13, 8))

    ax[0].plot(normalize(cars)[-len(data1):], color="#4D4D4D", linewidth=2)

    ax[0].set_ylabel("CARS Intensity (a.u.)")
    ax[0].grid(True, linestyle="--", alpha=0.5)
    ax[0].tick_params(axis='x')
    ax[0].tick_params(axis='y')

    ax[1].plot(normalize(raman)[-len(data1):], color="#d62728", linewidth=2, label='Raman',  alpha=0.9)

    ax[1].plot(normalize(dmd_reconstructed)[-len(data1):], color="#0072B2", linewidth=2, label='DMD', alpha=0.7)

    ax[1].set_xlabel("Wavenumber ($cm^{-1}$)")
    ax[1].set_ylabel("Raman Intensity (a.u.)")
    ax[1].grid(True, linestyle="--", alpha=0.5)
    ax[1].tick_params(axis='x')
    ax[1].tick_params(axis='y')

    plt.tight_layout()
    plt.legend(frameon=False)
    plt.show()

