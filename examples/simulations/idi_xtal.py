from __future__ import division

import os
import warnings
warnings.filterwarnings("ignore")

import sys
from time import time
import pylab as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore') 

from scipy.stats import binned_statistic_dd
from scipy.spatial import distance, cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from collections import Counter

sys.path.append("../..")
import bornagain as ba
from bornagain.target import crystal, Place
from bornagain.units import hc, keV
from bornagain.utils import vec_norm
from bornagain.simulate.clcore import ClCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import h5py

import pyopencl as cl
import pyopencl.array as clarray

gpu = True


def get_context_queue():
#   list the platforms
    platforms = cl.get_platforms()
    print("Found platforms (will use first listed):", platforms)
#   select the gpu
    my_gpu = platforms[0].get_devices(
        device_type=cl.device_type.GPU)
    assert( my_gpu)
    print("Found GPU(s):", my_gpu)
#   create the context for the gpu, and the corresponding queue
    context = cl.Context(devices=my_gpu)
    queue = cl.CommandQueue(context)
    return context, queue


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look\
     like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))



def sparse_idi(J, k_vecs, 
        qbins, Nq_waxs):
    
    idx = np.where(J)[0]
    dists =  distance.cdist( 
        k_vecs[idx], k_vecs[idx] )
    digs = np.digitize(dists, qbins) - 1
    digs = digs.ravel()
    
    Js = J[idx]
    
    weights = np.outer( Js, Js ) .ravel()
    
    good = digs<Nq_waxs

    H = np.bincount( digs[good], minlength=Nq_waxs , 
        weights=weights[good])

    return H


def sparse_idi_gpu( J, k_vecs, qbins, Nq_waxs,  context, queue):

    idx = np.where( J)[0]
    k_nonzero = k_vecs [ idx]
    J_nonzero = J [ idx]
    
    corrs = np.ascontiguousarray(np.zeros( (k_nonzero.shape[0], Nq_waxs), dtype=np.float32) )
    
    dq = qbins[1] - qbins[0]
    qmin = qbins.min()

    kernel = """ 
    __kernel void idi_corr(__global float* I_vecs,
                            __global float* k_vecs,
                             __global float* corrs,
                             int Nq, float dq, float qmin, int Nk){
    //  this is the unique ID of each worker, and each worker will be loading a single k vec
        
        int g_i = get_global_id(0);
        float I, I2, kx,ky,kz,kx2,ky2,kz2, q, dx, dy, dz;
        int bin;
        float _null;
    //  we pass 1D arrays to openCL, in row-major order
        
        I = I_vecs[g_i];
        
        kx = k_vecs[g_i*3];
        ky = k_vecs[g_i*3+1];
        kz = k_vecs[g_i*3+2];

        for(int i =0; i < Nk; i++){
            I2 = I_vecs[i];
            kx2 = k_vecs[i*3];
            ky2 = k_vecs[i*3+1];
            kz2 = k_vecs[i*3+2];
            dx=kx-kx2;
            dy=ky-ky2;
            dz=kz-kz2;
            q = sqrt( dx*dx+ dy*dy+ dz*dz); 
            bin = floor( (q - qmin ) / dq ) ; 
            if (bin < Nq)
                corrs[g_i*Nq + bin] += I*I2;
            else
                corrs[g_i*Nq + bin] += I*0;
        }
    }
    """
    #   setup opencl, compile bugs will show up here
    program = cl.Program(context, kernel).build()

#   move host arrays to GPU device, note forcing q_vecs and atom_vecs to be contiguous , ampsR and ampsI are already contiguous
    J_dev = clarray.to_device(queue, np.ascontiguousarray(J_nonzero.astype(np.float32)))
    k_dev = clarray.to_device(queue, np.ascontiguousarray(k_nonzero.astype(np.float32)))
    corrs_dev = clarray.to_device(queue, corrs)

#   specify scalar args (just tell openCL which kernel args are scalar)
    program.idi_corr.set_scalar_arg_dtypes(
            [None, None, None,np.int32, np.float32, np.float32, np.int32])
#   run the kernel
#   note there are 3 pre-arguments to our kernel, these are the queue, 
#   the total number of workers, and the desired worker-group size. 
#   Leaving worker-group size as None lets openCL decide a value (I think)
    program.idi_corr(queue, (k_nonzero.shape[0],), None, J_dev.data, k_dev.data,
        corrs_dev.data,  np.int32(Nq_waxs), np.float32( dq), np.float32( qmin), np.int32( k_nonzero.shape[0])   )

#   transfer data from device back to host
#    you can try to optimize enqueue_copy by passing different flags 
    cl.enqueue_copy(queue, corrs, corrs_dev.data)

    return np.sum(corrs,axis=0)

############################################################
# MAIN STARTS HERE
############################################################




# Viewing choices
qtview = True

# get GPU stuff
if gpu:
    context, queue = get_context_queue()

# save info
outdir = "idi_zinc"
if not os.path.exists(outdir):
    os.makedirs( outdir)
file_stride = 100
save_kvecs = "k_vecs_xtal"
save_normfactor="norm_factor_xtal"
print_stride=100
# output file names:
out_pre = "42-infinite_ps2_mor"
Waxs_file = os.path.join( outdir, "%s.Waxs"%out_pre)
Nshots_file = os.path.join( outdir, "%s.Nshots"%out_pre)

finite_photons = 1#True #False
dilute_limit = True

norm_factor =  None  # None #:x 1.#   None
# norm_factor = np.load( os.path.join( outdir , save_normfactor+".npy")) #"1-10mol_1modes_25x25x25unit/norm_factor_xtal.npy")
# this norm factor should correspond to your k_vecs, set as None to create, but it takes some time... 

#output waxs pattern
qmax_waxs = 1. #1. # inverse angstrom
Nq_waxs = 128

# How many diffraction patterns to simulate
n_patterns = 1000 # 12000
Num_modes = 1

# Intensity of the fluoress
photons_per_atom = 1 #00000000
n_unit_cell = 4

# whether to use Henke or Cromer mann
use_henke = True  # if False, then use Cromer-mann version

# Information about the object
n_molecules =1  #1# 10
box_size = 1000e-9
#box_size = 1000000000e-9
do_rotations = True #False
do_phases = True
do_translations = True

# Settings for pixel-array detector
n_pixels_per_dim = 128 # along a row
pixel_size = 0.0001 # meters, (small pixels that we will bin average later)
detector_distance = .1 # meter
block_size = 10,10

# Settings for spherical detector
spherical_detector = False #True
n_subdivisions = 3
radius = 1

####################################

# Information about the emission
photon_energy = 10.5 / keV
wavelength = hc / photon_energy # in meters
beam_vec = np.array([0, 0, 1.0]) # This shouldn't matter...

# Atomic positions of Mn atoms:
pdb_file = '../data/pdb/3wu2.pdb'
cryst = crystal.Molecule(pdb_file)
is_manga = cryst.Z==25
r = cryst.r[ is_manga]
r = r[:4] # take the first monomer in assymetric unit, 
cryst.lat.assemble(n_unit=n_unit_cell)
lattice = cryst.lat.vecs*1e-10
r = np.vstack([ r+l for l in lattice])
r -= r.mean(0)  # mean sub, I dunno it matters or not , but for rotations maybe...

r = np.load( 'xtal_zinc_8x8x8.npy' )*1e-10
r -= r.mean(0)  # mean sub, I dunno it matters or not , but for rotations maybe...

# maximum distance spanned by the molecule:
r_size = 5 * 25e-10 * np.sqrt(3)  
#r_size = cryst.a * n_unit_cell * np.sqrt(3) #distance.pdist(r).max()

n_atoms = r.shape[0]
print('Will simulate %d patterns' % (n_patterns))

if spherical_detector:
    detect_type = "SPHERE"
    print('Creating spherical detector...')
    ico = ba.detector.IcosphereGeometry(n_subdivisions=n_subdivisions, radius=radius)
    verts, faces, fcs = ico.compute_vertices_and_faces()
    n_faces = faces.shape[0]
    q = (2 * np.pi / wavelength) * fcs #(fcs - beam_vec)
    print('%d pixels' % (q.shape[0]))
else:
    detect_type='BOX'
    n = n_pixels_per_dim # shortcut
    p = pixel_size # shortcut
    
    pad = ba.detector.PADGeometry()
    pad.n_fs = n
    pad.n_ss = n
    pad.fs_vec = [p,0,0]
    pad.ss_vec = [0,p,0]
    pad.t_vec = [ (-n*p+p)*.5 , (-n*p+p)*.5, detector_distance]
    
    q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    pad_sh= pad.shape()
    print("Made the pad")
    print(pad_sh)
    #   combine the qs into a single vector...
    
    k_vecs = pad.position_vecs()
    k_vecs = vec_norm( k_vecs) * 2 * np.pi / wavelength
    
    q12_max = distance.euclidean( k_vecs[0], k_vecs[-1]  )
    q12_min = distance.euclidean( k_vecs[0], k_vecs[1])
    if save_kvecs is not None:
        np.save(os.path.join( outdir, save_kvecs), k_vecs)
    print("The pads cover the range %.4f to %.4f \
        inverse angstrom"%(q12_min*1e-10, q12_max*1e-10))
    print("Making solid angles...")
    sangs = np.abs(pad.solid_angles() )
    SA_frac = sangs.sum() / 4 / np.pi
    print ("solid angle fraction : %f" %SA_frac)

Npix = k_vecs.shape[0]
print("Simulating intensities for %d pixels \
    in the %s detector.." %(Npix, detect_type))

clcore = ClCore(group_size=1, double_precision=True)
q_dev = clcore.to_device(q)
seconds = 0
t0 = t=  time()

clcore.init_amps( Npix)

qbins = np.linspace( 0, qmax_waxs*1e10,  Nq_waxs+1)


if norm_factor is None:
    
    # make normalization factor
    if gpu:
         norm_factor = sparse_idi_gpu(np.ones( k_vecs.shape[0]), k_vecs, qbins, Nq_waxs, context, queue)
    
    else:
        # doing it this way to save on RAM
        norm_factor = np.zeros( Nq_waxs)

        for ik, kval in enumerate(k_vecs):
            kdists = distance.cdist( [kval], k_vecs )
            kdigs = np.digitize( kdists, qbins) -1
           
            C = Counter( kdigs.ravel() )
            counts = np.array( [ C[i_] for i_ in xrange( Nq_waxs)] )

            norm_factor += counts
            if ik%print_stride==0:
                print ( "Making norm factor: %d pixels remain..."\
                    % ( len(k_vecs) - ik))

    if save_normfactor:
        np.save( os.path.join( outdir, save_normfactor ), \
            norm_factor)




temp_waxs, temp_Nshots = [],[]
waxs = np.zeros(Nq_waxs)

def sample_I( I, SA_frac, photons_per_atom, 
            total_atoms, Num_modes  ):
    
    if I.dtype==np.float32:
        I = I.astype(np.float64)
    
    N_photons_measured =  int( SA_frac * \
            photons_per_atom * total_atoms / Num_modes)
    J = np.random.multinomial( N_photons_measured , I / I.sum() )

    return J

for pattern_num in range(0, n_patterns):

    if do_translations and not dilute_limit:
        placer = Place( box_edge=box_size, min_dist=r_size) 
        for i_n in range( n_molecules):
            placer.insert()
        Ts = placer.data 
    else:
        Ts = np.zeros([n_molecules, 3])
    
    rs = []
    for n in range(n_molecules):
        if do_rotations:
            R = ba.utils.random_rotation()
        else:
            R = np.eye(3)
        T = Ts[n, :]
        rs.append(np.dot(R, r.T).T + T)
    
    total_atoms = n_molecules*n_atoms
    print("Pattern %d / %d : Using %d atoms"\
        %(pattern_num, n_patterns , total_atoms))

    if finite_photons:
        J = np.zeros( Npix)
    else:
        J_inf = np.zeros( Npix)
    
    for _ in range( Num_modes):
        print("SImulating intensities")
        if dilute_limit:
            for r_mol in rs:
                phases = np.random.random(len(r_mol)) * 2 * np.pi
                fs = np.exp(1j * phases)
                        
                A = clcore.phase_factor_qrf(q_dev, r_mol, fs) #, q_is_qdev=True)
                #clcore.phase_factor_qrf_chunk(q_dev, r_mol, fs, Nchunk=2, q_is_qdev=True)
                #clcore.phase_factor_qrf_inplace(q_dev, r_mol, fs, q_is_qdev=True)
                #A = clcore.release_amps( reset=True)
                
                I_mol = np.abs(A) ** 2
                if finite_photons:
                    J += sample_I(I_mol, SA_frac, 
                        photons_per_atom, 
                        total_atoms, Num_modes)
                else:
                    J_inf += I_mol

        else: 
            #for r_mol in rs:
                #A = clcore.phase_factor_qrf(q_dev, r_mol, fs) #, q_is_qdev=True)
                #clcore.phase_factor_qrf_inplace(q_dev, r_mol, fs , q_is_qdev=True)
                #clcore.phase_factor_qrf_chunk(q_dev, r_mol, fs ,Nchunk=10, q_is_qdev=True)
            
            r_all = np.vstack( rs)
            phases = np.random.random(len(r_all)) * 2 * np.pi
            fs = np.exp(1j * phases)
            A = clcore.phase_factor_qrf(q_dev, r_mol, fs) #, q_is_qdev=True)
            
            #A = clcore.release_amps(reset=True)
            I = np.abs(A) ** 2
            if finite_photons:
                J += sample_I(I, SA_frac, 
                    photons_per_atom, total_atoms, 
                    Num_modes)
            else:
                J_inf += I
        

    print("Correlating")
    if finite_photons:
        if gpu:
            h = sparse_idi_gpu(J, k_vecs, qbins, Nq_waxs, context, queue)
        else:
            h = sparse_idi(J, k_vecs, qbins, Nq_waxs)#, context, queue)
    else:
        if gpu:
            h = sparse_idi_gpu(J_inf, k_vecs, qbins, Nq_waxs, context, queue)
        else:
            h = sparse_idi(J_inf, k_vecs, qbins, Nq_waxs) #, context, queue)
   
    #plt.imshow( J.reshape( pad_sh))
    #plt.show()

    waxs += h

    if pattern_num % file_stride == 0:
        waxs_norm = waxs / norm_factor
        temp_waxs.append(waxs_norm / waxs_norm[0] )
        temp_Nshots.append(pattern_num)
        np.save( os.path.join( outdir, 
            "temp_waxs_%d"%pattern_num) , 
            waxs_norm / waxs_norm[0])
        np.save( os.path.join( outdir, 
            "temp_Nshots_%d"%pattern_num) , 
            pattern_num)

    dt = time()-t
    if np.floor(dt) >= 3:
        t = time()
        sys.stdout.write('Pattern %6d of %6d ; \
            %3.0f%% ; %3.3g patterns/second\n' %
            (pattern_num, n_patterns, \
                100*pattern_num/float(n_patterns),
                pattern_num/(time() - t0)))    

# last save point:
waxs_norm = waxs / norm_factor
temp_waxs.append(waxs_norm / waxs_norm[0] )
temp_Nshots.append(pattern_num)
np.save( os.path.join( outdir, "temp_waxs_%d"%pattern_num) , waxs_norm / waxs_norm[0])
np.save( os.path.join( outdir, "temp_Nshots_%d"%pattern_num) , pattern_num)

np.save(Nshots_file, temp_Nshots)
np.save(Waxs_file, temp_waxs)
print("Saved np binary files %s.npy and %s.npy " %(Nshots_file, Waxs_file))

