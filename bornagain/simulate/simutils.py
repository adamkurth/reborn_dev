from __future__ import (absolute_import, division, print_function, unicode_literals)

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


def area_beam(lb, r):
    '''
    This function returns the cross sectional area of illuminated
    solvent assuming no crystal is present. if the beam side length
    is greater than 2r it reutrns the cross-sectional area of the jet.

    arguments
    ------------
    lb: the side length of the beam
    r:  the radius of the jet

    returns
    ------------
    float value of the cross sectional area of illuminated jet
    '''

    if lb >= 2 * r:
        return np.pi * r * r
    elif lb > 0 and lb < 2 * r:
        x = np.sqrt(4 * r * r - lb * lb) / 2
        theta = np.arccos(lb / (2 * r))
        sliver = (theta * r * r) - (x * lb / 2)
        return np.pi * r * r - 2 * sliver
    else:
        print("beam side length must be greater than zero.")


def area_crys(lc, r):
    '''
    This function finds the exposed cross-sectional area of solvent
    from a cubic crystal and cylindrical jet. If the crystal is larger
    than the beam it returns zero. Its mainly used in the function
    "volume_solvent."

    arguments
    -------------
    lc: the side length of the crystal
    r:  the radius of the jet

    returns
    -------------
    a float value for the cross-sectional area of illuminated solvent
    '''

    if lc >= 2 * r:
        return 0
    elif lc > np.sqrt(2) * r:
        x = np.sqrt(4 * r * r - lc * lc) / 2
        theta = np.arccos(lc / (2 * r))
        sliver = (theta * r * r) - (x * lc / 2)
        return 4 * sliver
    elif lc > 0 and lc <= np.sqrt(2) * r:
        return np.pi * r * r - lc * lc
    else:
        print("crystal side length must be greater than zero.")


def volume_solvent(lb, lc, r):
    '''
    This function returns the volume of illuminated solvent assuming
    a "square" beam, a cubic crystal and a cylindrical jet of solvent.

    arguments
    --------------
    lb: The side length of the square beam
    lc: The side length of the cubuc crystal
    r:  The radius of the solvent jet

    returns
    --------------
    float value of the volume of illuminated solvent
    '''

    if lb >= 2 * r:
        if lc >= lb:
            return 0
        elif lc >= 2 * r:
            return np.pi * r * r * (lb - lc)
        elif lc > np.sqrt(2) * r:
            return np.pi * r * r * (lb - lc) + lc * (area_crys(lc, r))
        elif lc > 0:
            return np.pi * r * r * (lb) - lc**3
        else:
            print("Crystal side length must be greater than zero")
    elif lb > 0:
        if lc >= 2 * r:
            return 0
        elif lc >= np.sqrt(2) * r:
            if lc >= lb:
                x = np.sqrt(4 * r * r - lc * lc)
                if lb >= x:
                    return area_crys(lc, r) * lb / 2
                elif lb < x:
                    return (area_beam(lb, r) - (lb * lc)) * lb
            if lb > lc:
                return (area_crys(lc, r) +
                        area_beam(lb, r) - np.pi * r * r) * lb
        elif lc > 0:
            if lc >= lb:
                return (area_beam(lb, r) - lb * lc) * lb
            else:
                return area_beam(lb, r) * lb - lc**3
        else:
            print("Crystal side length must be greater than zero.")
