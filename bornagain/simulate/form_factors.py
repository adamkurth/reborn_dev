from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np


def sphere_form_factor(radius, q_mags, check_divide_by_zero=True):

    """

    Form factor for a sphere of given radius, at given q magnitudes.  Assumes the scattering density is unity inside.

    Formula from Table A.1 in Guinier "X-ray diffraction in crystals, imperfect crystals, and amorphous bodies".  Rick
    confirms there are no approximations in this formula, and radius is correct (no it's not diameter by mistake).

    Args:
        radius (float): In SI units of course
        q_mags (numpy array): Also in SI units.
        check_divide_by_zero (bool): Check for divide by zero.  True by default.

    Returns: numpy array

    """

    qR = q_mags*radius
    amp = 4*np.pi*radius**3*(np.sin(qR)-qR*np.cos(qR))/qR**3
    if check_divide_by_zero is True:
        if any(q_mags == 0):
            w = np.where(q_mags == 0)
            if len(w) > 0:
                amp[w] = 4*np.pi*radius**3
    return amp
