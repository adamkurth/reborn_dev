r"""
Miscellaneous optimization and fitting routines.
"""

import numpy as np
from numpy.linalg import eig, inv
from .. import utils, detector


def fit_ellipse(x, y):
    r"""
    Given lists of coordinates :math:`(x,\;y)`, do least-squares fit to the coefficients :math:`a_i` of the function

    .. math::

        a_{xx} x^2 + a_{xy} xy + a_{yy}y^2 +a_x x + a_y y + a_1 = 0 \;.

    The algorithm ensures that the coefficients satisfy the ellipse condition :math:`4a_{xx} a_{yy}−a_{xy}^2 > 0`.
    We use the exact code found `here <http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html>`_, which in turn
    follows the approch in Fitzgibbon, A.W., Pilu, M., and Fischer R.B., *Direct least
    squares fitting of ellipsees*, Proc. of the 13th Internation Conference on Pattern Recognition, pp 253–257, Vienna,
    1996.

    For convenience, an alternative parameterization is also returned.  In this parameterization the ellipse is
    specified by the coordinates :math:`X, Y` and the following relations:
    
    .. math::
        \frac{X}{a^2} + \frac{Y}{b^2} = 1

    and

    .. math::
        x = \phantom{-}(X - X_0)\cos\theta + (Y - Y_0)\sin\theta \\
        y = -(X - X_0)\sin\theta + (Y - Y_0)\cos\theta

    By definition, :math:`a \ge  b` so that :math:`a` is the semi-major axis, and :math:`\theta` is the angle by which
    the semi-major axis is rotated.

    Args:
        x (|ndarray|): The :math:`x` coordinates
        y (|ndarray|): The :math:`y` coordinates

    Returns:
        |ndarray| with the coefficients :math:`[a_x, a_{xy}, a_{yy}, a_x, a_y, a_1, a, b, X_0, Y_0, \theta]`
    """

    x = np.double(x[:, np.newaxis])
    y = np.double(y[:, np.newaxis])
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = 2
    C[2, 0] = 2
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    params = V[:, n]  # These are the polynomial coefficients

    # Here we calculate the other parameterization
    a, b, c, d, e, f = params  # a[0], a[1], a[2], a[3], a[4], a[5]
    cond = b*b-4*a*c
    if cond < 0:
        smaj = - np.sqrt(2*(a*e*e+c*d*d-b*d*e+cond*f)*((a+c)+np.sqrt((a-c)**2+b*b)))/cond
        smin = - np.sqrt(2*(a*e*e+c*d*d-b*d*e+cond*f)*((a+c)-np.sqrt((a-c)**2+b*b)))/cond
        x0 = (2*c*d-b*e)/cond
        y0 = (2*a*e-b*d)/cond
        if (b == 0) and (a < c):
            tilt_angle = 0
        elif (b == 0) and (a > c):
            tilt_angle = 90
        else:
            tilt_angle = np.arctan((c-a-np.sqrt((a-c)**2+b*b))/b)
        params2 = np.array([smaj, smin, x0, y0, tilt_angle])
    else:
        raise ValueError('Your Ellipse is ill-conditioned.')
    
    params = np.concatenate([params, params2])
   
    return params


def ellipse_center(a):
    r"""
    Find the center :math:`x_0, \; y_0` of the ellipse function

    .. math::

        \frac{(x-x_0)^2}{a^2} + \frac{(y-y_0)^2}{b^2} = 1

    given the coefficients :math:`a_i` of the function

    .. math::

        a_1 + a_x x^2 + a_{xy} xy + a_{yy}y^2 +a_x x + a_y y = 0 \;.

    We use the exact code found `here <http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html>`_.  This function
    should typically be used along with the function :func:`fit_ellipse`.

    Args:
        a (|ndarray|): The array of :math:`a_i` coefficients

    Returns:
        |ndarray| with center position :math:`[x_0, \; y_0]`
    """
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0 = (c*d-b*f)/num
    y0 = (a*f-b*d)/num
    return np.array([x0, y0])


def ellipse_parameters(a):
    r"""
    Convert between the ellipse parameterization

    .. math::
        a_{xx} x^2 + a_{xy} xy + a_{yy}y^2 +a_x x + a_y y + a_1 = 0 \;.

    and the parameterization specified by the coordinates :math:`X, Y` and the following relations:

    .. math::
        \frac{X}{a^2} + \frac{Y}{b^2} = 1

    and

    .. math:
        x = \phantom{-}(X - X_0)\cos\theta + (Y - Y_0)\sin\theta \\
        y = -(X - X_0)\sin\theta + (Y - Y_0)\cos\theta

    By definition, :math:`a \ge b` so that :math:`a` is the semi-major axis, and :math:`\theta` is the angle by which
    the semi-major axis is rotated.

    Arguments:
        a (|ndarray|): The coefficients :math:`[a_x, a_{xy}, a_{yy}, a_x, a_y, a_1]`

    Returns:
        |ndarray|: The coefficients :math:`[a, b, X_0, Y_0, \theta]`

    """
    a, b, c, d, e, f = a[0], a[1], a[2], a[3], a[4], a[5]
    cond = b*b-4*a*c
    if cond < 0:
        smaj = - np.sqrt(2*(a*e*e+c*d*d-b*d*e+cond*f)*((a+c)+np.sqrt((a-c)**2+b*b)))/cond
        smin = - np.sqrt(2*(a*e*e+c*d*d-b*d*e+cond*f)*((a+c)-np.sqrt((a-c)**2+b*b)))/cond
        x0 = (2*c*d-b*e)/cond
        y0 = (2*a*e-b*d)/cond
        if (b == 0) and (a < c):
                tilt_angle = 0
        elif (b == 0) and (a > c):
                tilt_angle = 90
        else:
            tilt_angle = np.arctan((c-a-np.sqrt((a-c)**2+b*b))/b)
        return np.array([smaj, smin, x0, y0, tilt_angle])
    else:
        return print("This method probably won't work for you!")


def fit_ellipse_pad(pad_geometry, data, threshold, mask=None):
    r"""
    Fit an ellipse to pixels above threshold.  In order to deal with the possibility of multiple PADs with different
    detector distances, the x,y coordinates are projected onto a plane located at a distance of 1 meter from the origin.
    The x-ray beam is assumed to be along the z direction (we can change this if the need arises).

    Args:
        pad_geometry (list of |PADGeometry|'s): PAD geometry.
        data (list of |ndarray|'s): Data to threshold.
        threshold (float): Threshold value.  The x,y coordinates from pixels above this will be used in the fit.
        mask (list of |ndarray|'s):

    Returns:
        |ndarray| : Ellipse fit parameters (see :func:`fit_ellipse <reborn.analysis.optimize.fit_ellipse>` function)
    """
    pads = utils.ensure_list(pad_geometry)
    data = utils.ensure_list(data)
    if mask is not None:
        mask = utils.ensure_list(mask)
    else:
        mask = [np.ones(d.shape) for d in data]
    fit_mask = []
    for i in range(len(pads)):
        m = np.zeros(data[i].shape)
        m[data[i] >= threshold] = 1
        fit_mask.append(m*mask[i])
    px = np.concatenate([p.position_vecs()[:, 0].ravel() / p.position_vecs()[:, 2].ravel() for p in pads])
    py = np.concatenate([p.position_vecs()[:, 1].ravel() / p.position_vecs()[:, 2].ravel() for p in pads])
    fit_mask = detector.concat_pad_data(fit_mask)
    fit_params = fit_ellipse(px[fit_mask == 1], py[fit_mask == 1])
    return fit_params
