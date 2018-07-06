import h5py
from joblib import Parallel, delayed
import numpy as np
from scipy.spatial import distance
import glob

#~~~~~~~~~~~~~~~~~~~~~~~~
# definitions 
n_jobs = 8
fnames = glob.glob("6-10mol_4mode/6-10mol_4mode_pattern*.npy")
print("Found %d files"% len(fnames))
k_vecs = np.load("k_vecs.npy")
#norm_factor = None
norm_factor = np.load("norm_factor.npy")
# output file names:
out_pre = "modes/6-10mol_4modes"

# params
max_files = 490000
Nq = 512
phot_per_mol = 4
file_stride = 100
Nmol = 10
qmax = 12 # inverse angstrom
print_stride = 100
Nmodes = 4
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

def main( files_per_mode, jid, out_pre):
    Waxs_file = "%s.Waxs"%out_pre
    Nshots_file = "%s.Nshots"%out_pre
    temp_waxs, temp_Nshots = [],[]
    waxs = np.zeros(Nq)
    for i,f in enumerate(files_per_mode):
        J = np.zeros( Npixels)
        Is = np.load(f)
        if len(Is.shape) > 1:
            for I in Is:
                J += np.random.multinomial( phot_per_mol*Nmol / len(Is) , I / I.sum() )
        else:
            J += np.random.multinomial( phot_per_mol*Nmol  , Is / Is.sum() )
        h = sparse_idi(J)
        waxs += h
        
        if i%file_stride == 0 and i !=0:
            #waxs_norm = waxs/ norm_factor
            temp_waxs.append(waxs.copy() )
            temp_Nshots.append(i)
        if i%print_stride==0:
            print ("JOB %d: %d shots remain..."%( jid, len(files_per_mode)-i))
        
    Nshots_file = Nshots_file+"_job%d"%jid
    np.save (Nshots_file, temp_Nshots)
    Waxs_file =Waxs_file+"_job%d"%jid 
    np.save( Waxs_file, temp_waxs)
    print("JID %d: Saved np binary files %s.npy and %s.npy " %(jid, Nshots_file, Waxs_file))
    return Nshots_file, Waxs_file 

files_per_mode = fnames[:max_files]
files_split = np.array_split( files_per_mode, n_jobs)

results = Parallel( n_jobs=n_jobs)( delayed(main) ( files_split[jid], jid, out_pre) \
    for jid in xrange(n_jobs) )

nshots_all = []
waxs_all = []
for r in results:
    nshots, waxs = np.load(r[0]+".npy"), np.load( r[1]+".npy")
    nshots_all.append(  nshots)
    waxs_all.append( waxs)

with h5py.File( out_pre+".h5py", 'w') as h5:
    
    waxs_dset = []
    n_dset = []
    for waxs, nshots in zip( zip(*waxs_all), zip(*nshots_all) ):
        n_dset.append( np.sum( nshots) )
        
        waxs_sum = np.sum( waxs, axis=0)
        waxs_norm = waxs_sum / norm_factor
        waxs_dset.append( waxs_norm / waxs_norm[0])
    h5.create_dataset("waxs", data=waxs_dset)
    h5.create_dataset("nshots", data=n_dset)


