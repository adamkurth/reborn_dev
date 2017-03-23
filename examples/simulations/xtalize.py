import time

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array

import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.simulate.refdata as refdata
import bornagain.target.crystal as crystal
#import refdata

def sphericalize(lattice):
    """attempt to sphericalize a 2D lattice point array"""
    center = lattice.mean(0)
    rads = np.sqrt( np.sum( (lattice - center)**2,1))
    max_rad = min( lattice.max(0))/2.
    return lattice [ rads < max_rad ] 

def amplitudes_with_cmans(q, r, Z):
    """
    compute scattering amplitudes 
    =============================
    q: 2D np.array of q vectors 
    r: 2D np.array of atom coors
    Z: 1D np.array of atomic numbers corresponding to r
    """
    cman = refdata.get_cromermann_parameters(Z) # 
    form_facts = refdata.get_cmann_form_factors(cman, q) 
    ff_mat = array( [form_facts[z] for z in cryst.Z]).T
    amps = np.dot( q, r.T)
    amps = np.exp( 1j*amps)
    amps = np.sum( amps*ff_mat, 1)
    return amps

def amplitudes(q, r):
    """
    compute scattering amplitudes without form factors
    ==================================================
    q: 2D np.array of q vectors 
    r: 2D np.array of atom coors
    """
    amps = np.dot( q, r.T)
    amps = np.exp( 1j*amps)
    amps = np.sum( amps,1)
    return amps

# Generate Q vectors
#~~~~~~~~~~~~~~~~~~~
pl = ba.detector.PanelList()
nPixels = 1001
pixelSize = 100e-6
detectorDistance = 0.5
wavelength = 1 #1e-10
si_energy = ba.units.hc/ (wavelength*1e-10)

pl.simple_setup(nPixels, nPixels+1, pixelSize, detectorDistance, wavelength)
q = pl[0].Q # q vectors!

# shape of 2D images
img_sh = (pl[0].nS, pl[0].nF)

# Load atom information
#~~~~~~~~~~~~~~~~~~~~~~
pdbFile = '/home/dermen/.local/ext/bornagain/examples/data/pdb/2LYZ.pdb'  
cryst = crystal.structure(pdbFile)
cman = refdata.get_cromermann_parameters(cryst.Z) # 
form_facts = refdata.get_cmann_form_factors(cman, q) # form factors!
r = cryst.r*1e10 # atom positions!

lookup = {}
form_facts_arr = np.zeros( (len(form_facts), q.shape[0] )  )
for i,z in enumerate(form_facts):
    lookup[z] = i
    form_facts_arr[i] = form_facts[z]
atomID = np.array( [ lookup[z] for z in cryst.Z] ).astype(np.int32)

# setup the open CL stuff
#~~~~~~~~~~~~~~~~~~~~~~~~
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
context = cl.Context(devices=my_gpu_devices)
queue = clcore.queue
group_size = clcore.group_size


# Molecular transform
A_mol = clcore.phase_factor_qrf2(q, r, form_facts_arr, atomID, R=None, 
    context=context, queue=queue,group_size=group_size)

img_mol = (np.abs(A_mol)**2).reshape(img_sh)


################################
################################

# Lattice transformation

#  unit cell edges
a = np.array([79.1, 0, 0]) # angstroms
b = np.array([0, 79.1, 0])
c = np.array([0, 0, 37.9])

n_unit = 50
# lattice coordinates
r_lattice = np.array( [ i*a + j*b + k*c 
    for i in xrange( n_unit) 
        for j in xrange( n_unit) 
            for k in xrange(n_unit) ] )


# sphericalize the lattice.. 
r_lattice = sphericalize( r_lattice)

# make dummie form factors to use bornagain... 
f_dummie = np.ones_like( r_lattice , dtype=np.complex)

print("Lattice transforming %d atoms"%r_lattice.shape[0])
t = time.time()
A_lat = clcore.phase_factor_qrf(q, 
    r_lattice, 
    f_dummie, 
    R=None, 
    context=context, 
    queue=queue,
    group_size=group_size)
print("Took %.4f sec"%(time.time() - t))

img_lat = (np.abs(A_lat)**2).reshape(img_sh)

# Full image 
img = (np.abs(A_lat*A_mol)**2).reshape(img_sh)

