import numpy as np
from numpy.linalg import eig, inv


def fit_ellipse(x, y):
    r"""
    Given lists of coordinates :math:`(x,\;y)`, do least-squares fit to the coefficients :math:`a_i` of the function

    .. math::

        a_x x^2 + a_{xy} xy + a_{yy}y^2 +a_x x + a_y y + a_1 = 0 \;.

    The algorithm ensures that the coefficients satisfy the ellipse condition :math:`4a_{x,x} a_{y,y}−a_{x,y}^2 > 0`.
    We use the exact code found `here <http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html>`_, which in turn
    follows the approch by Fitzgibbon, Pilu and Fischer in Fitzgibbon, A.W., Pilu, M., and Fischer R.B., *Direct least
    squares fitting of ellipsees*, Proc. of the 13th Internation Conference on Pattern Recognition, pp 253–257, Vienna,
    1996.

    Args:
        x (numpy array): The :math:`x` coordinates
        y (numpy array): The :math:`y` coordinates

    Returns:
        numpy array with the coefficients :math:`[a_x, a_{xy}, a_{yy}, a_x, a_y, a_1]`
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
    a = V[:, n]
    return a


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
        a (numpy array): The array of :math:`a_i` coefficients

    Returns:
        numpy array with center position :math:`[x_0, \; y_0]`
    """
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0 = (c*d-b*f)/num
    y0 = (a*f-b*d)/num
    return np.array([x0, y0])
