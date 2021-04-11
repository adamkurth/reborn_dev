import reborn
import matplotlib.pyplot as plt
import numpy as np

# Prototype for making comer mann atomic density maps.

# Define the density map grid.  Always follow the convention in the reborn docs.
x_max = 5e-10
x_min = -x_max
n_x = 1000
dx = (x_max - x_min)/(n_x-1)

# Below corresponds to n_x = 4, with atom position r = 0.75
# * is bin center, | is the bin edge, X is the atom location
#
# |_____*_____|__X__*_____|_____*_____|_____*_____|


# Construct atom radial profiles.  We are going to
prof_r_max = 10e-10
prof_n = 1000
prof_dr = prof_r_max/(prof_n-1)
prof_r = np.arange(prof_n)*prof_dr
profs = reborn.target.atoms.get_cromer_mann_densities(np.arange(89)+1, prof_r)
r_max = 5e-10  # max radius

dens = np.zeros(n_x)  # The density map

rs = np.array([-1, +1])*1e-10
zs = np.array([1, 2])
fs = zs
n_atoms = rs.size

for atom_n in range(n_atoms):
    z = zs[atom_n]
    r = rs[atom_n]
    f = fs[atom_n]  # scattering factor.  atom should integrate to this value
    prof = profs[z, :]
    r_idx = int(np.round((r - x_min) / dx))  # nearest index of atom
    n_subarray = int(np.ceil(r_max / dx))  # sub-array for building up and normalizing one-atom density
    sub_map = np.zeros(2 * n_subarray + 1)
    for i in np.arange(2 * n_subarray + 1):  # i is index in temp map
        gi = r_idx - n_subarray + i  # global index in map
        r_map = gi*dx + x_min  # coordinate at global index
        d = np.sqrt((r - r_map)**2)  # distance from this pixel to atom
        prof_idx = d / prof_dr  # Index in the radial profile
        pir = int(np.floor(prof_idx))  # Index rounded down
        if pir+1 > prof_n:  # Skip when out of range of provided radial profile
            continue
        val = prof[pir] * (1 - (prof_idx - pir)) + prof[pir + 1] * (prof_idx - pir)  # Linear interpolation
        sub_map[i] = val
    sub_map = sub_map * f / np.sum(sub_map)  # Normalize the sampled atom density
    for i in np.arange(2 * n_subarray + 1):  # i is index in sub array
        gi = r_idx - n_subarray + i  # global index in map
        dens[gi % n_x] += sub_map[i]  # Note the modulus here - we wrap around when positions are out of bounds

x = np.arange(n_x)*dx + x_min
plt.plot(x*1e10, dens, '.')
plt.xlabel('Position (angstrom)')
plt.show()
