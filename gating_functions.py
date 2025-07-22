## Imports

import sys, os
from pathlib import Path

parent_folder = str(Path.cwd().parents[0])
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from sigpy import mri
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
import twixtools
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from admm.utils_moco import stacked_nufft_operator,golden_angle_2d_readout, golden_angle_coords_3d, pocs, phase_based_gating_peak_to_peak, phase_based_gating, create_gates


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks, medfilt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


def phase_based_gating_updated(signal, num_gates, order=15):
    """ phase-based gating; distribute data between adjacent maxima and minima evenly to the gates
    NOTE: this only works if all mins and max can be found reliably."""

    ## 1. Find relative minima and maxima, find whichever array is shorter
    maxima = np.asarray(argrelextrema(signal, np.greater, order=order)).squeeze()   
    minima = np.asarray(argrelextrema(signal, np.less, order=order)).squeeze()
    common_length = np.min([len(maxima), len(minima)])  ##Use whichever array is shorter

    ## Compare the two only up to that length
    maxima = maxima[:common_length]
    minima = minima[:common_length]

    ## Sanity check
    if maxima[0] < minima[0]:
        if not(np.all(maxima[:-1] < minima[1:])):
            raise Exception("Error during gating- missed max or min")
    else:
        if not (np.all(minima[:-1] < maxima[1:])):
            raise Exception("Error during gating - missed max or min")
        
    ## 2. Create empty array where we will alternate min and max values
    raw_idx = np.zeros((len(maxima) + len(minima)), dtype=np.int32).squeeze()

    if maxima[0] < minima[0]:
        phase='exp'
        raw_idx[0::2] = maxima
        raw_idx[1::2] = minima
    else:
        phase='insp'
        raw_idx[0::2] = minima
        raw_idx[1::2] = maxima
    
    ## 3. Create empty array to assign gating indicies for the entire respiratory signal
    idx = np.zeros_like(signal, dtype=np.int32)

    ## 4. Handle signal before first extrema
    ## Exp. is last gate. If we start with exp, number samples before first extrema from num_gates to 0
    ## If we start with insp, number samples before first extrema from 0 to num_gates
    if raw_idx[0] > 0:
        if phase == 'exp':
            d = np.round(np.linspace(num_gates, 1, raw_idx[0]))
        else:
            d = np.round(np.linspace(1, num_gates, raw_idx[0]))

        idx[:raw_idx[0]] = d

    ## Handle signal between extrema
    for i in range(len(raw_idx)-1):
        ind_a = raw_idx[i]
        ind_b = raw_idx[i+1]
        if phase == 'insp':
            d = np.round(np.linspace(1, num_gates, (ind_b - ind_a + 1)))
            idx[ind_a:ind_b+1] = d    
            phase = 'exp'
        else: ##exp
            d = np.round(np.linspace(num_gates, 1, (ind_b-ind_a+1)))
            idx[ind_a:ind_b+1] = d
            phase = 'insp'

    ## Handle signal at end after last extrema
    if raw_idx[-1] < len(signal)-1:
        if phase=='insp':
            d = np.round(np.linspace(1, num_gates, len(signal)-raw_idx[-1]))
        else:
            d = np.round(np.linspace(num_gates, 1, len(signal)-raw_idx[-1]))
        idx[raw_idx[-1]:]=d
        
    return idx




# # Integration with existing code
# def gate_resp_signal_improved(ksp_data, resp_signal, num_gates, img_shape, spokes_to_discard=500, gating_method='adaptive_hybrid'):
#     """
#     Improved version of your gate_resp_signal function
#     """
#     # divide phase between two peaks of the signal evenly.
#     # Thus, in- and expiration go into different gates.
#     resp_trimmed = resp_signal[spokes_to_discard:]
   
#     num_coils, num_slices, num_spokes, num_samples = ksp_data.shape
    
#     # Use improved gating instead of phase_based_gating_peak_to_peak
#     idx = robust_respiratory_gating(resp_trimmed, num_gates, method=gating_method)
    
#     print(f"First 100 gate assignments: {idx[0:100]}")
#     print(f"Gate assignments 300-500: {idx[300:500]}")
    
#     ## divide kspace and spoke data:
#     coords = golden_angle_coords_3d(img_shape, num_spokes, num_samples)
#     ndims = coords.shape[-1]
    
#     ### Combine slice and spoke dimension so we have all temporal samples
#     kspace_temporal = np.reshape(ksp_data, (num_coils, num_slices*num_spokes, num_samples))
#     kspace_trimmed = kspace_temporal[:, spokes_to_discard:, :]
#     coords_temporal = np.reshape(coords, (num_slices*num_spokes, num_samples, ndims))
#     coords_trimmed = coords_temporal[spokes_to_discard:, :, :]
    
#     ### Check shape
#     print(f'kspace_trimmed.shape = {kspace_trimmed.shape}')
#     print(f'coords_trimmed.shape = {coords_trimmed.shape}')
    
#     data_bins, spoke_bins = create_gates_updated(kspace_trimmed, coords_trimmed, idx, num_gates)
#     for i in range(num_gates):
#         data_bins[i], spoke_bins[i] = reshape_gate(data_bins[i], spoke_bins[i], num_coils, num_slices, num_samples, ndims)
#         print(f"Gate {i+1}: kspace shape = {data_bins[i].shape}, coords shape = {spoke_bins[i].shape}")
    
#     return idx, resp_trimmed, data_bins, spoke_bins