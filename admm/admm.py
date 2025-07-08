import os

import numpy as np
import cupy as cp
import json


import sigpy
import sigpy as sp

from copy import deepcopy
#sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])

def admm_mr(ds, Fs, img_shape, motion_est_fun, motion_inv_fun, motion_parms, rho, beta, target_gate_index, output_dir, device, do_pre_initialization=True,num_iter=15,motion_base='zu_lam'):
    """"
    Joint Reconstruction of Motion and Image Data using ADMM
    
    Parameters
    ----------
    ds: list 
        kspace data
    Fs: list
        Fourier ops for ds
    img_shape: tuple
        shape of the image(s)
    motion_est_fun: func
        Function to estimate motion
    motion_inv_fun: func
        Function to invert motion
    motion_parms:
        Parameters for motion_est_fun,
    rho: float
        rho value for ADMM
    beta: float
        smoothing parameter
    target: int
        reference phase index
    output_dir: string
        directory to save results to
    device: int
        device for calculations (CPU/GPU)
    do_pre_initialization: bool
        initialize z_k, lambda, S_k
    num_iter: int
        number of ADMM outer iterations:
    motion_base: string
        how to estimate motion:
            'zu_lam': align (z_k + u_k) and lambda
            'z':      align (z_k) and (z_target)

    """
   
    # Conventions:
    # 1) the gate dimension comes first in all ndarrays [followed by spatial (x,y,z) [and by vector component dim for motion vector fields]]
    # 1a)      For storage, though, the gate dim has to be moved to the last position (for image viewers)
    # 2) All fields are cupy, except motion vector fields are numpy (for now)
    # 2a) all motion functions will take cupy fields and convert them to numpy
    # 2b) motion fields are returned as numpy
    
    # write parms to output dir:
    with open(os.path.join(output_dir,'parm.json'),'w') as f:
        json.dump( {'rho':rho,'beta':beta, \
                 'target_gate_index':target_gate_index,'motion_est_fun':motion_est_fun.__name__,\
                    'motion_parms':motion_parms,'do_pre_initialization':do_pre_initialization}\
                        ,f )
        
    # help function to save ndarrays
    def save_data(data,filename,iter_num):
            tmp = cp.asnumpy(data)
            # move gate axis to last position for storage:
            if len(tmp.shape)>3:
                    tmp = np.moveaxis(tmp.copy(),0,-1)

            with open(os.path.join(output_dir,filename+'_{:03d}_abs.v'.format(iter_num)),'wb') as f:
                f.write(np.reshape(np.abs(tmp),-1,order='F').astype(np.float32))
            with open(os.path.join(output_dir,filename+'_{:03d}_compl.v'.format(iter_num)),'wb') as f:
                f.write(np.reshape(tmp,-1,order='F').astype(np.complex64))

    # help function to create interpolation operators:
    def motion_vec_field_2_op_list(mvf,m):
    
        xp = np

        # ensure border of MVF is all zeros:
        if True:
            mvf[:,0,...]=0
            mvf[:,-1,...]=0
            mvf[:,:,0,...]=0
            mvf[:,:,-1,...]=0
            mvf[:,:,:,0,...]=0
            mvf[:,:,:,-1,...]=0

        # asserting mvf is 4d, with time/gate dimension in first dim:
        op_list=[]
        in_x = xp.arange(0,m[0],1)
        in_y = xp.arange(0,m[1],1)
        in_z = xp.arange(0,m[2],1)
        x,y,z = xp.meshgrid(in_x,in_y,in_z,indexing='ij')
        base_grid = xp.stack((x,y,z),axis=-1)
        for gate in range(mvf.shape[0]):
            op_list.append(sp.linop.Interpolate(tuple(m),base_grid+mvf[gate,...]))

        return op_list

    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    # independent recons to init. z's and estimate intial motion fields
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
   

    num_gates = len(ds)
   
    
    do_wo_moco_recon=False
    
    # number of PDHG iterations for ADMM subproblem (2)
    max_num_iter_subproblem_2 = 100
    sigma_pdhg = 1.0

    # bool whether to solve the original or approximated subproblem (2)
    use_subproblem2_approx = True

    # random seed
    seed = 1
    np.random.seed(seed)

    

    with cp.cuda.Device(device):
        # gradient operator, factor in front makes sure that |G| = 1
        G = sigpy.linop.Gradient(img_shape)

        # normalize the norm of the gradient operator
        max_eig_G = sigpy.app.MaxEig(G.H * G, dtype=cp.complex64, max_iter=30).run()
        G = (1 / np.sqrt(max_eig_G)) * G

        # prox for TV prior
        proxg_ind = sigpy.prox.L1Reg(G.oshape, beta / num_gates)

        ind_recons = cp.zeros((num_gates, *img_shape), dtype=cp.complex64)

        if do_pre_initialization:
            for i in range(num_gates):
                alg01 = sigpy.app.LinearLeastSquares(Fs[i],
                                                    sigpy.to_device(ds[i],device),
                                                    G=G,
                                                    proxg=proxg_ind,
                                                    max_iter=500,
                                                    sigma=sigma_pdhg)
                ind_recons[i, ...] = alg01.run()
                del alg01

            save_data(ind_recons,'indep_recons',0)

            #--------------------------------------------------------------------
            #--------------------------------------------------------------------
            # initial estimate of motion fields (operators)
            #--------------------------------------------------------------------
            #--------------------------------------------------------------------

            
            # i) estimate motion
            tmp_mvf = motion_est_fun(cp.abs(ind_recons[target_gate_index,...]),
                                       cp.abs(ind_recons),
                                       motion_parms,target_gate_index)

            # ii) invert estimated field:
            tmp_mvf_inv = motion_inv_fun(tmp_mvf)
            
            # iii) create sigpy interpolation ops:
            Ss = motion_vec_field_2_op_list(tmp_mvf,img_shape)
            Ss_inv = motion_vec_field_2_op_list(tmp_mvf_inv, img_shape)
            
            # move gate axis to last position for storage
            with open(os.path.join(output_dir,'S_indep_recons_inv.mvf'),'wb') as f:
                f.write(np.reshape(np.moveaxis(tmp_mvf_inv,0,-1),-1,order='F').astype(np.float32))
            with open(os.path.join(output_dir,'S_indep_recon.mvf'),'wb') as f:
                f.write(np.reshape(np.moveaxis(tmp_mvf,0,-1),-1,order='F').astype(np.float32))
            
            del tmp_mvf, tmp_mvf_inv



        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # reconstruction of all the data without motion modeling as reference
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        if do_wo_moco_recon:
            proxg_sum = sigpy.prox.L1Reg(G.oshape, beta)

            alg0 = sigpy.app.LinearLeastSquares(sigpy.linop.Vstack(Fs),
                                                sigpy.to_device(cp.concatenate([x.ravel() for x in ds]),device),
                                                G=G,
                                                proxg=proxg_sum,
                                                max_iter=500,
                                                sigma=sigma_pdhg)
            recon_wo_moco = alg0.run()
            del alg0

            save_data(recon_wo_moco,'non_mc_recon',0)

        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # ADMM
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        
        # prox for subproblem 2 - note extra (1/rho) which is needed for subproblem 2
        proxg2 = sigpy.prox.L1Reg(G.oshape, beta / rho)
        # prox for subproblem 2 - note extra (1/rho) which is needed for the approximate subproblem 2
        proxg2a = sigpy.prox.L1Reg(G.oshape, beta / (num_gates * rho))

        # initialize all variables
        
        if do_pre_initialization:
            lam = ind_recons[target_gate_index, ...].copy()
            zs = ind_recons.copy()
            us = cp.zeros_like(zs)
            # for i in range(num_gates):
            #     us[i] = us[i] + zs[i] - Ss[i](lam)
        else:
            # init lambda, z, S, S^-1 with zeros
            lam = cp.zeros(img_shape,dtype=cp.complex64)
            zs = cp.zeros((num_gates, *img_shape), dtype=cp.complex64)
            #Ss_tmp = np.zeros((*img_shape,3,num_gates))
            #Ss = mvf_array_2_op_list(Ss_tmp,img_shape)
            Ss = motion_vec_field_2_op_list(np.zeros((num_gates,*img_shape,3)), img_shape)

            
            #Ss_inv = mvf_array_2_op_list(Ss_inv_tmp,img_shape)
            Ss_inv = motion_vec_field_2_op_list(np.zeros((num_gates,*img_shape,3)), img_shape)
            us = cp.zeros_like(zs)
            print('after no init:')
            print(zs.shape)
            print(len(Ss))
            #print(S[0].shape)
        cost = np.zeros(num_iter)

        #recons = np.zeros((num_iter, *img_shape), dtype=cp.complex64)


        for i_outer in range(num_iter):
            ###################################################################
            # subproblem (1) - data fidelity + quadratic - update for z1 and z2
            ###################################################################

            for i in range(num_gates):
                alg11 = sigpy.app.LinearLeastSquares(Fs[i],
                                                    sigpy.to_device(ds[i],device),
                                                    x=zs[i],
                                                    lamda=rho,
                                                    z=(Ss[i](lam) - us[i, ...]))
                zs[i, ...] = alg11.run()
                del alg11
            save_data(zs,'z',i_outer)
            ###################################################################
            # subproblem (2) - optimize lambda
            ###################################################################
            # MF:
            # invert Ss
            if use_subproblem2_approx:
                # optimize approximation of subproblem (2) using the inverse
                # of the motion deformation operators
                v = cp.zeros_like(lam)
                for i in range(num_gates):
                    ############################################################
                    # invert deformation operator here
                    ############################################################

                    # for "simple" circular shift, the inverse equal the adjoint
                    # !!! not true for a non-rigid deformation operator !!!
                    #S_inv = Ss[i].H
                    # MF:
                    # Ss_inv is defined as list of Ops just like Ss:
                    v += Ss_inv[i](us[i] + zs[i])

                    

                v /= num_gates

                if i_outer == 0:
                    pdhg_u2a = cp.zeros(G.oshape, dtype=lam.dtype)

                alg2a = sigpy.alg.PrimalDualHybridGradient(
                    proxfc=sigpy.prox.Conj(proxg2a),
                    proxg=sigpy.prox.L2Reg(img_shape, 1, y=v),
                    A=G,
                    AH=G.H,
                    x=deepcopy(lam),
                    u=pdhg_u2a,
                    tau=1 / sigma_pdhg,
                    sigma=sigma_pdhg)

                for _ in range(max_num_iter_subproblem_2):
                    alg2a.update()

                lam = alg2a.x
            else:
                # optimize the exact subproblem (2) which requires knowledge of the
                # adjoint of the motion deformation operators

                Ss = sigpy.linop.Vstack(Ss)
                y = (us + zs).ravel()

                # we could call LinearLeastSquares directly, but we will use call the
                # PHDG directly which allows us to store the dual variable of PDHG
                # for warm start of the following iteration

                # run PDHG to solve subproblem (2)
                A = sigpy.linop.Vstack([Ss, G])
                proxfc = sigpy.prox.Stack(
                    [sigpy.prox.L2Reg(y.shape, 1, y=-y),
                    sigpy.prox.Conj(proxg2)])

                if i_outer == 0:
                    max_eig = sigpy.app.MaxEig(A.H * A, dtype=y.dtype,
                                            max_iter=30).run()
                    pdhg_u = cp.zeros(A.oshape, dtype=y.dtype)

                alg2 = sigpy.alg.PrimalDualHybridGradient(
                    proxfc=proxfc,
                    proxg=sigpy.prox.NoOp(A.ishape),
                    A=A,
                    AH=A.H,
                    x=deepcopy(lam),
                    u=pdhg_u,
                    tau=1 / (max_eig * sigma_pdhg),
                    sigma=sigma_pdhg)

                for _ in range(max_num_iter_subproblem_2):
                    alg2.update()

                lam = alg2.x

            
            save_data(lam,'lambda',i_outer)
            


            

            ###################################################################
            # update of displacement fields (motion operators) based on z1, z2
            ###################################################################

           
            # motion_est_fun(target_data, gate_data, parms, skip_idx=-1)

            # i) estimate motion
            if motion_base=='zu_lam':   # align (lam) and (z_k+u_k):
                tmp_mvf = motion_est_fun( cp.abs(lam),
                                        cp.abs(zs+us),
                                        motion_parms,target_gate_index)
                
            elif motion_base=='z':      # align (z_target_gate_index) and (z_k)
                tmp_mvf = motion_est_fun(cp.abs(zs[target_gate_index,...]),
                                       cp.abs(zs),
                                       motion_parms,target_gate_index)
            else:
                raise ValueError('Parameter motion_base not set properly.')
            
            # ii) invert estimated field:
            tmp_mvf_inv = motion_inv_fun(tmp_mvf)
            
            # iii) create sigpy interpolation ops:
            Ss = motion_vec_field_2_op_list(tmp_mvf,img_shape)
            Ss_inv = motion_vec_field_2_op_list(tmp_mvf_inv, img_shape)
            
            # move gate axis to last position for storage
            with open(os.path.join(output_dir,'S_inv_{:03d}.v'.format(i_outer)),'wb') as f:
                f.write(np.reshape(np.moveaxis(tmp_mvf_inv,0,-1),-1,order='F').astype(np.float32))
            with open(os.path.join(output_dir,'S_{:03d}.v'.format(i_outer)),'wb') as f:
                f.write(np.reshape(np.moveaxis(tmp_mvf,0,-1),-1,order='F').astype(np.float32))
            
            del tmp_mvf, tmp_mvf_inv

            # If motion is estimated on the 'z+u' and 'lam', update of the 'u' should come 
            # after motion estimation:
            # if motion_base=='zu_lam':
            #     for i in range(num_gates):
            #         us[i] = us[i] + zs[i] - Ss[i](lam)
            #     save_data(us,'u',i_outer)
            
            ###################################################################
            # update of dual variables
            ###################################################################
            # If motion is estimated on the 'z' only, the 'u' should be updated before motion estimation:
            #if motion_base=='z':
            # Leave the u update here for now:
            for i in range(num_gates):
                us[i] = us[i] + zs[i] - Ss[i](lam)
            save_data(us,'u',i_outer)


            ###################################################################
            # evaluation of cost function
            ###################################################################
            

            # evaluate the cost function
            prior = float(G(lam).real.sum() + G(lam).imag.sum())

            data_fidelity = np.zeros(num_gates)

            for i in range(num_gates):
                e = Fs[i](Ss[i](lam)) - sigpy.to_device(ds[i],device)
                data_fidelity[i] = float(0.5 * (e.conj() * e).sum().real)

            cost[i_outer] = data_fidelity.sum() + beta * prior
            with open(os.path.join(output_dir,'cost.json'),'w') as f:
                json.dump({'cost':cost.tolist()},f)

          
        with open(os.path.join(output_dir,'cost.json'),'w') as f:
            json.dump({'cost': cost.tolist()},f) 

            



