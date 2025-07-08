from __future__ import annotations

import sigpy
import numpy as np
import numpy.typing as npt
import cupy.typing as cpt
from scipy.signal import argrelextrema

def golden_angle_2d_readout(kmax: float, num_spokes: int,
                            num_points: int) -> npt.NDArray:
    """2D golden angle kspace trajectory

    Parameters
    ----------
    kmax : float
        maximum absolute k-space value
    num_spokes : int
        number of spokes (readout lines)
    num_points : int
        number of readout points per spoke

    Returns
    -------
    npt.NDArray
    """
    tmp = np.linspace(-kmax, kmax, num_points)
    k = np.zeros((num_spokes, num_points, 2))

    ga = np.pi / ((1 + np.sqrt(5)) / 2)

    for i in range(num_spokes):
        phi = (i * ga) % (2 * np.pi)
        k[i, :, 0] = tmp * np.cos(phi)
        k[i, :, 1] = tmp * np.sin(phi)

    return k


def stacked_nufft_operator(
        img_shape: tuple,
        coords: npt.NDArray | cpt.NDArray) -> sigpy.linop.Diag:
    """setup a stacked 2D NUFFT sigpy operator acting on a 3D image
       the opeator first performs a 1D FFT along the "z" axis (0 or left-most axis)
       followed by applying 2D NUFFTS to all "slices"
       
    Parameters
    ----------
        img_shape: tuple
            shape of the image
        coords: (numpy or cupy) array 
            coordinates of the k-space samples
            shape (n_k_space_points,2)
            units: "unitless" -> -N/2 ... N/2 at Nyquist (sigpy convention)

    Returns
    -------
        Diag: a stack of NUFFT operators
    """

    # setup the FFT operator along the "z" axis
    ft0_op = sigpy.linop.FFT(img_shape, axes=(0, ))

    # setup a 2D NUFFT operator for the start
    nufft_op = sigpy.linop.NUFFT(img_shape[1:], coords)

    # reshaping operators that remove / expand dimensions
    rs_in = sigpy.linop.Reshape(img_shape[1:], (1, ) + img_shape[1:])
    rs_out = sigpy.linop.Reshape((1, ) + tuple(nufft_op.oshape),
                                 nufft_op.oshape)

    # setup a list of "n" 2D NUFFT operators
    ops = img_shape[0] * [rs_out * nufft_op * rs_in]

    # apply 2D NUFFTs to all "slices" using the sigpy Diag operator
    return sigpy.linop.Diag(ops, iaxis=0, oaxis=0) * ft0_op



def stacked_nufft_operator_sens(
        img_shape: tuple,
        coords: npt.NDArray | cpt.NDArray,
        mps: npt.NDArray | cpt.NDArray) -> sigpy.linop.Diag:
    """setup a stacked 2D NUFFT sigpy operator acting on a 3D image
       the opeator first performs a 1D FFT along the "z" axis (0 or left-most axis)
       followed by applying 2D NUFFTS to all "slices"
       
    Parameters
    ----------
        img_shape: tuple
            shape of the image
        coords: (numpy or cupy) array 
            coordinates of the k-space samples
            shape (n_k_space_points,2)
            units: "unitless" -> -N/2 ... N/2 at Nyquist (sigpy convention)
        mps: (numpy or cupy) array
            sensitivity maps of shape (num_channels, *img_shape)

    Returns
    -------
        Diag: a stack of NUFFT operators
    """

    num_channels = len(mps)

    ft0_op = sigpy.linop.FFT(img_shape, axes=(0, ))

    # setup a 2D NUFFT operator for the start
    nufft_op = sigpy.linop.NUFFT(img_shape[1:], coords)

    # reshaping operators that remove / expand dimensions
    rs_in = sigpy.linop.Reshape(img_shape[1:], (1, ) + img_shape[1:])
    rs_out = sigpy.linop.Reshape((1, ) + tuple(nufft_op.oshape),
                                    nufft_op.oshape)

    # setup a list of "n" 2D NUFFT operators
    ops = img_shape[0] * [rs_out * nufft_op * rs_in]

    # apply 2D NUFFTs to all "slices" using the sigpy Diag operator
    full_op= sigpy.linop.Diag(ops, iaxis=0, oaxis=0) * ft0_op


    
    #### Combine Sensitivity Op (mult with sens) and respective ft0+nuFFT op:

    #sensitivity = np.ones((num_channels,*img_shape),dtype=np.complex64)
    S = sigpy.linop.Multiply(img_shape,mps)

    rs_in_sense = sigpy.linop.Reshape(img_shape,(1,)+img_shape)
    rs_out_sense = sigpy.linop.Reshape((1,)+tuple(full_op.oshape),full_op.oshape)
    return  sigpy.linop.Diag(num_channels*[rs_out_sense*full_op*rs_in_sense],iaxis=0,oaxis=0)*S

def golden_angle_coords_3d(img_shape,num_spokes,num_points):
    # assuming isotropic in-plane resolution:
    coords_2d= golden_angle_2d_readout(img_shape[1]//2,num_spokes,num_points)
    return coords_3d_from_2(img_shape,coords_2d)
    shape_3d = [img_shape[0]]+list(coords_2d.shape)
    shape_3d[3]+=1
    # 3d spoke coords:
    coords_3d = np.zeros(shape_3d,dtype=coords_2d.dtype)
    slice_coords = np.linspace(-ksp_k[0]/2.,+ksp_k[0]/2.,img_shape[0])
    for i,slice in enumerate(range(img_shape[0])):
        coords_3d[i,:,:,1:]=coords_2d
        coords_3d[i,:,:,0]=slice_coords[i]

    return coords_3d

def coords_3d_from_2(img_shape, coords_2d):
    shape_3d = [img_shape[0]]+list(coords_2d.shape)
    shape_3d[3]+=1
    # 3d spoke coords:
    coords_3d = np.zeros(shape_3d,dtype=coords_2d.dtype)
    slice_coords = np.linspace(-img_shape[0]/2.,+img_shape[0]/2.,img_shape[0])
    for i,slice in enumerate(range(img_shape[0])):
        coords_3d[i,:,:,1:]=coords_2d
        coords_3d[i,:,:,0]=slice_coords[i]

    return coords_3d

def pocs(ksp,fourier_op,center_width,pf_index):
    # assumptions:
    # - first dim is partial fourier
    # - lower half of kspace is zero-filleld
    # pf_index is the first index in the pf-dim to contain data
    
    dimens = len(ksp.shape)
    cent_half = center_width//2

    # filter for the PF dimension
    tmp_filt = np.hanning(center_width+2)
    hann = np.zeros(ksp.shape[0])
    hann[ksp.shape[0]//2-cent_half:ksp.shape[0]//2+cent_half]=tmp_filt[1:-1]
     
    # low-pass filter kspace
    if dimens==3:
        #print('3d POCS')
        hamm_1 = np.hamming(ksp.shape[1])
        hamm_2 = np.hamming(ksp.shape[2])
        ksp_filt = ksp \
            *np.reshape(hann,(len(hann),1,1)) \
            *np.reshape(hamm_1,(1,len(hamm_1),1)) \
            *np.reshape(hamm_2,(1,1,len(hamm_2))) 
    elif dimens==2:    
        #print('2d POCS')    
        hamm = np.hamming(ksp.shape[1])
        ksp_filt = ksp \
            *np.reshape(hann,(len(hann),1)) \
            *np.reshape(hamm,(1,len(hamm))) 
    else:
        print(ksp.shape)
        raise(AssertionError('Could not determine dimensions'))
    # get phase estimation:
    #print('max ksp_filt: ' + str(np.max(np.abs(ksp_filt))))
    im_lr = fourier_op.H(ksp_filt)
    phase = np.exp(1j*np.angle(im_lr))
    #phase /=np.prod(ksp.shape)

    # inital guess for corrected image:
    #im_hr = fourier_op.H(np.conj(ksp))
    im_hr = fourier_op.H(ksp)
    im_hr = np.abs(im_hr)*phase
    #print('max ksp: ' + str(np.max(np.abs(ksp))))
    #print('mean ksp upper half: ' + str(np.mean(np.abs(ksp[-pf_index:,...]))))
    for iter in range(10):

        ksp_pocs = fourier_op(im_hr)
        #print('mean ksp_pocs uppcs half: ' + str(np.mean(np.abs(ksp_pocs[-pf_index:,...]))))
        #print('max ksp_pocs: ' + str(np.max(np.abs(ksp_pocs))))
        #print('relation ksp/ksp_pocs: ' + str( np.max(np.abs(ksp)) / np.max(np.abs(ksp_pocs))  ))
        # fill in known data:
        ksp_pocs[pf_index:,...]=ksp[pf_index:,...]
        
         
        # combine abs(im) ans phase:
        im_hr = fourier_op.H(ksp_pocs)
        im_hr = np.abs(im_hr)*phase

    return ksp_pocs

def phase_based_gating(signal, num_gates, order=15):
    """ phase-based gating; distribute data between adjacent maxima and minima evenly to the gates
        NOTE: this only works if all mins and max can be found reliably."""

    maxima = np.asarray(argrelextrema(signal,np.greater,order=order)).squeeze()
    minima = np.asarray(argrelextrema(signal,np.less,order=order)).squeeze()
    common_length = np.min([len(maxima), len(minima)])
    maxima = maxima[:common_length]
    minima = minima[:common_length]

    if maxima[0]<minima[0]:
        if not (np.all(maxima[:-1] < minima[1:])):
            raise Exception("error during gating - missed max or min")
    else:
        if not (np.all(minima[:-1] < maxima[1:])):
            raise Exception("error during gating - missed max or min")

    # join min and max values:
    raw_idx = np.zeros((len(maxima)+len(minima)),dtype=np.int32).squeeze()

    if maxima[0]<minima[0]:
        phase='exp'
        raw_idx[0::2]=maxima
        raw_idx[1::2]=minima
    else:
        phase='insp'
        raw_idx[0::2]=minima
        raw_idx[1::2]=maxima

    idx = np.zeros_like(signal,dtype=np.int32)
    
    for i in range(len(raw_idx)-1):
        ind_a = raw_idx[i]
        ind_b = raw_idx[i+1]
        if phase=='insp':
            d = np.round(np.linspace(1,num_gates,(ind_b-ind_a+1)))
            idx[ind_a:ind_b+1]=d
            # set next phase:
            phase='exp'
        else: # exp
            d = np.round(np.linspace(num_gates,1,(ind_b-ind_a+1)))
            idx[ind_a:ind_b+1]=d
            # set next phase:
            phase='insp'
    return idx

def phase_based_gating_peak_to_peak(signal, num_gates, order=15):
    """ Phase-based gating between peaks in the signal"""
    maxima = np.asarray(argrelextrema(signal,np.greater,order=order)).squeeze()

    idx = np.zeros_like(signal,dtype=np.int32)

    for i in range(len(maxima)-1):
        ind_a = maxima[i]
        ind_b = maxima[i+1]

        d = np.round(np.linspace(1,num_gates,(ind_b-ind_a+1)))
        idx[ind_a:ind_b+1]=d

    return idx

def create_gates(ksp,coords, idx,num_gates):
    data_bins=[]
    spoke_bins=[]

    for bin in range(1,num_gates+1):
        current_kspace = ksp[:,:,(idx==bin),...]
        data_bins.append(current_kspace)
        current_ks = coords[:,(idx==bin),...]
        spoke_bins.append(current_ks)

    return data_bins, spoke_bins
