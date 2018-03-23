import numpy as np

# import refdata


# def amplitudes_with_cmans(q, r, Z):
#     """
#     compute scattering amplitudes
#
#     q: 2D np.array of q vectors
#     r: 2D np.array of atom coors
#     Z: 1D np.array of atomic numbers corresponding to r
#
#     FIXME: Derek - is this used anywhere?
#     """
#
#     cman = refdata.get_cromermann_parameters(Z)
#     form_facts = refdata.get_cmann_form_factors(cman, q)
#     ff_mat = np.array([form_facts[z] for z in Z]).T.astype(np.float32)
#     amps = (np.dot(q, r.T)).astype(np.float32)
#     amps = np.exp(1j * amps).astype(np.complex64)
#     amps = np.sum(amps * ff_mat, 1).astype(np.complex64)
#     return amps
#
#
# def amplitudes(q, r):
#     """
#     compute scattering amplitudes without form factors
#
#     q: 2D np.array of q vectors
#     r: 2D np.array of atom coors
#
#     FIXME: Derek - is this used anywhere?
#     """
#     amps = np.dot(q, r.T)
#     amps = np.exp(1j * amps)
#     amps = np.sum(amps, 1)
#     return amps


def sphericalize(lattice):
    """attempt to sphericalize a 2D lattice point array"""
    center = lattice.mean(0)
    rads = np.sqrt(np.sum((lattice - center) ** 2, 1))
    max_rad = min(lattice.max(0)) / 2.
    return lattice[rads < max_rad]