## Imports

import sys, os
from pathlib import Path

parent_folder = str(Path.cwd().parents[0])
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from sigpy import mri
import time
import pickle
from sklearn.decomposition import PCA
import sigpy as sp
import cupy as cp
import numpy as np
import twixtools
import matplotlib.pyplot as plt
from sigpy.mri.app import TotalVariationRecon
from scipy.signal import butter,filtfilt
from gating_functions import phase_based_gating_updated
from admm.utils_moco import stacked_nufft_operator,golden_angle_2d_readout, golden_angle_coords_3d, pocs, phase_based_gating_peak_to_peak, phase_based_gating, create_gates


def get_ksp_from_twix(data_file):
    multi_twix = twixtools.read_twix(str(data_file))
    mapped = twixtools.map_twix(multi_twix)
    data_0 = mapped[0]['image']
    print(data_0.non_singleton_dims)
    data_0.flags['remove_os']=True

    echo_num=0                                                  # first echo is spoke data
    num_points = int(mapped[0]['hdr']['Config']['NImageLins'])  # number of points on one spoke
    kspace_0 = data_0[...,echo_num,0,0,0,:,0,0,:,:,:num_points]
    kspace_0 = kspace_0.squeeze()
    kspace_0 = np.transpose(kspace_0,(2,0,1,3))
    return mapped, kspace_0


def butter_lowpass_filter(data, cutoff_hz, fs_hz, order=2):
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_resp_signal(ksp_data, mapped):
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape

    TR = float(mapped[0]['hdr']['Config']['TR'])/1000000. ## microseconds to seconds
    print(f"TR from data: {TR} sec")

    n_samples = num_spokes * num_slices  # total time points/temporal samples

    # Sampling frequency in Hz
    fs = 1/(TR)
    print(f"Sampling frequency (Hz): {fs:.3f}")

    cutoff = 0.25  # respiratory frequency in Hz (~15 bpm)

    # Sanity check: cutoff must be less than Nyquist
    if cutoff >= fs / 2:
        cutoff = 0.49 * fs
        print(f"Cutoff frequency adjusted to: {cutoff:.3f} Hz to stay below Nyquist")

    # Build full time scale in SECONDS
    full_time_scale_s = np.arange(n_samples) * TR  # time in s
    print(f'full timescale goes to {full_time_scale_s[-1]}')

    signal_all_coils = []
    for c in range(num_coils):
        # mean magnitude over samples 130:140, shape (num_slices, num_spokes)
        signal = np.mean(np.abs(ksp_data[c, :, :, 130:140]), axis=-1)
        signal = signal.T.flatten()  # flatten to 1D (time points)

        filtered_signal = butter_lowpass_filter(signal, cutoff, fs, order=2)
        signal_all_coils.append(filtered_signal)

    coil_matrix = np.stack(signal_all_coils, axis=0)  # shape: (coils, time points)

    print(f"coil matrix shape: {coil_matrix.shape}")

    # plot raw and filtered signal from last coil as example (time in seconds)
    plt.figure(figsize=(12, 4))
    plt.plot(full_time_scale_s, signal, label="Raw Signal (Last Coil)")
    plt.plot(full_time_scale_s, filtered_signal, label="Filtered Signal (Last Coil)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal strength per excitation")
    plt.title("Raw and Filtered Signal (Last Coil)")
    plt.legend()
    # plt.xlim(0, 60)  # show first 60 seconds
    plt.show()

    return coil_matrix, TR

def perform_pca(coil_matrix, n_components=1):
    # coil_matrix shape: (coils, time_points)
    data_for_pca = coil_matrix.T  # shape: (time_points, coils)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(data_for_pca)
    resp_signal = pcs[:, 0]
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return pcs, pca.explained_variance_ratio_, resp_signal

def plot_resp_signal(pcs, TR):
    n_samples = pcs.shape[0]
    time_ms = np.arange(n_samples) * TR
    plt.figure(figsize=(12, 4))
    plt.plot(time_ms, pcs[:, 0], label="Respiratory Signal (PCA)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Respiratory Signal from PCA")
    plt.legend()
    plt.xlim(0, 60)
    plt.show()


def create_gates_updated(ksp,coords, idx,num_gates):
    data_bins=[]
    spoke_bins=[]

    for bin in range(1,num_gates+1):
        current_kspace = ksp[:,(idx==bin),:]
        data_bins.append(current_kspace)
        current_ks = coords[(idx==bin),...]
        spoke_bins.append(current_ks)

    return data_bins, spoke_bins

def reshape_gate(data_bin, spoke_bin, num_coils, num_slices, num_samples, ndims):
    N = data_bin.shape[1]
    max_valid = (N // num_slices) * num_slices
    
    data_bin = data_bin[:, :max_valid, :]
    spoke_bin = spoke_bin[:max_valid, :, :]

    num_spokes_per_slice = max_valid // num_slices
    data_bin = data_bin.reshape(num_coils, num_slices, num_spokes_per_slice, num_samples)
    spoke_bin = spoke_bin.reshape(num_slices, num_spokes_per_slice, num_samples, ndims)

    return data_bin, spoke_bin

def gate_resp_signal(ksp_data, resp_signal, num_gates, img_shape, spokes_to_discard=700):
    # Do not use the transient part of the signal (the large overshoot) for gating
    resp_trimmed = resp_signal[spokes_to_discard:]
    
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape
    idx = phase_based_gating_updated(resp_trimmed, num_gates, order=25)

    ## divide kspace and spoke data:
    img_shape = (58,256,256) 
    coords = golden_angle_coords_3d(img_shape,num_spokes,num_samples)
    ndims = coords.shape[-1]

    ### Combine slice and spoke dimension so we have all temporal samples
    kspace_temporal = np.reshape(ksp_data, (num_coils, num_slices*num_spokes, num_samples))
    kspace_trimmed = kspace_temporal[:, spokes_to_discard:, :]
    coords_temporal = np.reshape(coords, (num_slices*num_spokes, num_samples, ndims))
    coords_trimmed = coords_temporal[spokes_to_discard:, :, :]

    ### Check shape
    # print(f'kspace_trimmed.shape = {kspace_trimmed.shape}')
    # print(f'coords_trimmed.shape = {coords_trimmed.shape}')

    data_bins, spoke_bins = create_gates_updated(kspace_trimmed, coords_trimmed, idx, num_gates)
    for i in range(num_gates):
        data_bins[i], spoke_bins[i] = reshape_gate(data_bins[i], spoke_bins[i], num_coils, num_slices, num_samples, ndims)
        print(f"Gate {i}: kspace shape = {data_bins[i].shape}, coords shape = {spoke_bins[i].shape}")

    return idx, resp_trimmed, data_bins, spoke_bins


def visualize_resp_gating(resp_signal, idx, TR, num_gates, title="Respiratory Gating Visual"):
    ## 1. Create time axis
    time_s = np.arange(len(resp_signal))*TR

    fig = plt.figure(figsize=(15, 10))

    ## Define colors for gates
    gate_colors = plt.cm.Set3(np.linspace(0,1, num_gates))

    ## Plot 1: Signal with color coded gates
    plt.plot(time_s, resp_signal, 'k-', linewidth=0.8, alpha=0.7, label="Respiratory Signal")

    for i in range(1,(num_gates+1)):
        mask = (idx==i)
        if np.any(mask):
            plt.scatter(time_s[mask], resp_signal[mask],
                        c=[gate_colors[i-1]], s=20, alpha=0.8, label=f'Gate {i}', edgecolors='none')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Respiratory Signal Amplitude')
    plt.title(f'{title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()


def run_tv_param_sweep_single_gate(data_bins, spoke_bins, mps, img_shape, 
                                   lambda_values, gate_idx=0, max_iter=100, device=0, save_results=True, output_dir=None):
    
    '''Run TotalVariationRecon with multiple lambda values for a single respiratory gate
    
    
    Params
    ----------------------
    data_bins : list of ndarrays
        Gated ksp data for each resp gate, shape (channels, slices, spokes, samples). Spokes should be only varying 
    spoke_bins : list of ndarrays
        Coordinate arrays for each resp gate, shape (slices, spokes, samples, ndim). ndim = 3
    mps: array
        Sensitivity maps of length = number of channels
    img_shape : tuple
        Final desired image shape, (z, y, x)
    lambda_values = list
        Lambda values to sweep, regularization parameter
    gate_idx : int
        Gate to use for the reconstruction
    max_iter : int  
        Number of iterations to run
    device: int
        GPU device to use
    save_results : bool
        True to save results
    output_dir : Path/string


    Output
    -----------------
    results: dict
        Dictionary containing recons and metadata

    '''

    ## Set output directory for final results
    if output_dir is None:
        output_dir = Path('tv_sweep_results')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)

    ## Initialize results storage
    results = {
        'lambda_values': lambda_values,
        'img_shape': img_shape,
        'max_iter': max_iter,
        'reconstructions':{},
        'timing':{},
        'parameters':{}
    }

    total_recons = len(lambda_values)
    current_recon = 0


    ## Initial output statements
    print(f'Starting param sweep for gate {gate_idx}')
    print(f'Lambda values = {lambda_values}')
    print(f'Using {max_iter} iters per recon')
    print(f'Total reconstructions: {total_recons}')
    print(f'Results saved to: {output_dir}')
    print("-" * 60)

    for lam in lambda_values:
        print(f'\n Testing lambda = {lam}')

        ## Initialize storage for this lambda
        results['reconstructions'][lam] = None
        results['timing'][lam] = None

        ## Test with coordinates
        start_time = time.time()

        try:
            alg_tv = TotalVariationRecon(data_bins[gate_idx], mps,
                                         lam, coord=spoke_bins[gate_idx], device=device)
            result_lam = alg_tv.run()
            recon_lam = cp.asnumpy(result_lam)

            time_lam = time.time() - start_time

            ## Set results in array
            results['reconstructions'][lam] = recon_lam
            results['timing'][lam] = time_lam

            print(f'    Completed in {time_lam}')

        except Exception as e:
            print(f'    Failed: {str(e)}')
            results['reconstructions'][lam] = None
            results['timing'][lam] = None

        current_recon +=1

    if save_results:
        results_file = output_dir/f'tv_sweep_gate{gate_idx}_lambda{len(lambda_values)}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f'\nResults saved to {results_file}')

    return results


if __name__ == '__main__':
    ## 1. Load data
    data_path = Path('/home/lilianae/data/NaF_MtSinai/')
    data_file = data_path /'anon_meas_MID00118_FID60738_Tho_fl3d_star_vibe_991_nav_tj_2000sp_AllCoils_SOS.dat'

    ksp_mapped, ksp_data = get_ksp_from_twix(data_file) ## Shape (ncoils, nslices, nspokes, nsamples)
    print(f'\n  ksp_data.shape = {ksp_data.shape} = (coils, slices, spokes, samples) ')


    ## 2. Get respiratory signal
    coil_matrix, TR = get_resp_signal(ksp_data, ksp_mapped)
    pcs, _, resp_signal = perform_pca(coil_matrix, n_components=1)


    ## 3. Gate k-space data
    ncoils, nslices, nspokes, nsamples = ksp_data.shape
    num_gates = 3
    img_shape = (nslices, nsamples, nsamples)
    idx, resp_trimmed, data_bins, spoke_bins = gate_resp_signal(ksp_data, resp_signal, num_gates=num_gates, img_shape=img_shape)
    

    ## 4. TV Reconstruction
    ## Set coil sens maps to be uniform 1s for initial tests
    mps_shape = (ncoils, *img_shape)
    mps = np.ones(mps_shape, dtype=np.complex64)

    gate_idx_to_reconstruct = 1
    lambda_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    final_recon = run_tv_param_sweep_single_gate(data_bins, spoke_bins, mps, img_shape, 
                                                 lambda_values=lambda_values, gate_idx=gate_idx_to_reconstruct)
