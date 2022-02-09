import reborn
import matplotlib.pyplot as plt
import numpy as np

# try:
#     import density_f
# except ImportError:
#     from numpy.f2py import compile
#     f90file = "density.f90"
#     compile(source=open(f90file, "rb").read(),
#             modulename=f90file.replace(".f90", "_f"),
#             extension=".f90", #extra_args='--f90flags="-fcheck=all -Wall"',
#             verbose=False)
#     import density_f

density_f = reborn.fortran.import_f90('density.f90', hash=True)

from time import time


def make_cromer_mann_density_map(x_max = 5e-10, x_min = -5e-10,
                                 n_bin = 1000, zs = np.array([1, 2]),
                                 z_loc = np.array([-1, +1])*1e-10,
                                 r_max = 5e-10):
    r"""
    Make Cromer-Mann atomic density maps.

    Coordinates refer to the centers of bins, not the edges.

    Example: n_bin = 4 and atom position r = 0.75
             * is the bin center
             | is the bin edge
             X is the atom location
    
    |_____*_____|__X__*_____|_____*_____|_____*_____|

    Arguments:
        x_max (float): maximum value along x
                       default = 5e-10
        x_min (float): minimum value along x
                       default = -5e-10
        n_bin (int): number of bins along x
                     default = 1000
        zs (numpy.ndarray): atoms
                            default = np.array([1, 2])
        z_loc (numpy.ndarray): atom locations
                               default = np.array([-1, +1])*1e-10
        r_max (float): neighborhood of the atom
                       default = 5e-10

    Returns:
        density_map (numpy.ndarray): Comer-Mann atomic density map computed by python
        density_map_fortran (numpy.ndarray): Comer-Mann atomic density map computed by fortran
        tf (float): duration of fortran code in s
        tp (float): duration of python code in s
    """
    t = time()
    n_atoms = zs.size
    # Define the density map grid.
    dx = (x_max - x_min) / (n_bin - 1)
    # Construct atom radial profiles.
    prof_r_max = 10e-10
    n_profs = 1000
    prof_dr = prof_r_max / (n_profs - 1)
    prof_r = np.arange(n_profs) * prof_dr
    cromer_mann_densities = reborn.target.atoms.get_cromer_mann_densities(np.arange(89) + 1, prof_r)
    n_subarray = int(np.ceil(r_max / dx))  # sub-array for building up and normalizing one-atom density

    # Crommer-Mann density map
    density_map = np.zeros(n_bin)  # python
    density_map_fortran = np.zeros(n_bin)  # fortran
    density_map_fortran[:] = density_f.density1d(x_max, x_min, n_bin,
                                                 n_atoms, n_profs, n_subarray,
                                                 zs, z_loc, cromer_mann_densities.T,
                                                 prof_dr)
    tf = time() - t

    fs = zs

    for atom_n in range(n_atoms):
        z = zs[atom_n]  # pick out atom
        r = z_loc[atom_n]  # get atom location
        f = fs[atom_n]  # scattering factor. atom should integrate to this value
        prof = cromer_mann_densities[z, :]  # get radial profile of atom (Crommer-Mann density)
        r_indx = int(np.round((r - x_min) / dx)) + 1  # nearest index (atom can be anywhere in the bin)
        # instead of calculating the atom density over the whole domain
        # we only integrate the density in the neighborhood of the atom
        sub_map = np.zeros(2 * n_subarray + 1)  # neighborhood density map of the atom
        for i in np.arange(2 * n_subarray + 1):  # i is index in temp map
            global_indx = r_indx - n_subarray + i  # global index in density_map
            r_map = global_indx * dx + x_min  # coordinate at global index
            d = np.sqrt((r - r_map)**2)  # distance from this pixel to atom
            prof_indx = d / prof_dr  # Index in the radial profile
            pir = int(np.floor(prof_indx))  # Index rounded down
            if pir + 1 > n_profs:  # Skip when out of range of provided radial profile
                continue
            val = prof[pir] * (1 - (prof_indx - pir)) + prof[pir + 1] * (prof_indx - pir)  # Linear interpolation
            sub_map[i] = val
        sub_map = sub_map * f / np.sum(sub_map)  # Normalize the sampled atom density
        for i in np.arange(2 * n_subarray + 1):  # i is index in sub array
            global_indx = r_indx - n_subarray + i  # global index in map
            density_map[global_indx % n_bin] += sub_map[i]  # Note the modulus here - we wrap around when positions are out of bounds
    tp = time() - t
    return density_map, density_map_fortran, tf, tp

if __name__ == "__main__":
    x_max = 5e-10
    x_min = -5e-10
    n_bin = 1000
    dx = (x_max - x_min) / (n_bin - 1)
    dens, dens_f, tf, tp = make_cromer_mann_density_map(x_max, x_min, n_bin)

    x = np.arange(n_bin)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
    ax[0].plot(x, dens, ".", label="Python (%.3f s)" % tp)
    ax[0].plot(x, dens_f, 'x', label="Fortran (%.3f s)" % tf)
    ax[0].set_xlabel("Position (angstrom)")
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title("Atomic Density")
    ax[1].plot(x, dens - dens_f, ".")
    ax[1].set_xlabel("Position (angstrom)")
    ax[1].grid()
    ax[1].set_title("Error")
    plt.show()
