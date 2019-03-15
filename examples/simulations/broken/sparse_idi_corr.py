import numpy as np
from scipy.spatial import distance
import glob

#~~~~~~~~~~~~~~~~~~~~~~~~
# definitions 
fnames = glob.glob("5-10mol/5-10mol_pattern*.npy")
print("Found %d files"% len(fnames))
k_vecs = np.load("k_vecs.npy")
#norm_factor = None
norm_factor = np.load("norm_factor.npy")
# output file names:
out_pre = "2modes"
Waxs_file = "%s.Waxs"%out_pre
Nshots_file = "%s.Nshots"%out_pre

# params
max_files = 5000
Nq = 512
phot_per_mol = 4
file_stride = 500
Nmol = 10
qmax = 12 # inverse angstrom
print_stride=100
Nmodes = 2
Npixels = k_vecs.shape[0]
#~~~~~~~~~~~~

qbins = np.linspace( 0, qmax * 1e10, Nq+1)
H = np.zeros(Nq)
if norm_factor is None:
    # make normalization factor
    # doing it this way to save on RAM
    norm_factor = np.zeros( Nq)
    for ik,kval in enumerate(k):
        kdists = distance.cdist( [kval], k )
        kdigs = np.digitize( kdists, qbins)-1
        norm_factor += np.bincount( kdigs.ravel(), minlength=Nq)
        if ik%print_stride==0:
            print ( "%d pixels remain..."% ( len(k) - ik))

def sparse_idi(J):
    """
    these uses only correlates the non-zero photon count pixels
    for massive speedups

    This is a crucial contribution to the analysis pipeline
    """
    idx = np.where(J)[0]
    dists =  distance.cdist( k_vecs[idx], k_vecs[idx] )
    digs = np.digitize(dists, qbins) - 1
    Js = J[idx]
    weights = np.outer( Js, Js ) 
    H = np.bincount( digs.ravel(), minlength=Nq , weights=weights.ravel())
    return H

temp_waxs, temp_Nshots = [],[]
waxs = np.zeros(Nq)
files_per_mode = np.array_split( fnames[:max_files], len(fnames[:max_files])/Nmodes  )
for i,fs in enumerate(files_per_mode):
    J = np.zeros( Npixels)
    for f in fs:
        I = np.load(f).astype(np.float64)
        J += np.random.multinomial( phot_per_mol*Nmol / Nmodes , I / I.sum() )
    h = sparse_idi(J)
    waxs += h
    if i%file_stride == 0:
        waxs_norm = waxs/ norm_factor
        temp_waxs.append(waxs_norm / waxs_norm[0] )
        temp_Nshots.append(i)
    if i%print_stride==0:
        print ("%d shots remain..."%( len(files_per_mode)-i))
    
np.save(Nshots_file, temp_Nshots)
np.save(Waxs_file, temp_waxs)
print("Saved np binary files %s.npy and %s.npy " %(Nshots_file, Waxs_file))



