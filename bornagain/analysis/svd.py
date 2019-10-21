import numpy as np
from scipy.linalg import orth, qr
from scipy.sparse.linalg import svds


def addblock_svd_update(U, S, V, A, force_orth=False):

    '''

    This is a nearly direct translation of the Matlab code found here:
    https://pcc.byu.edu/scripts/addblock_svd_update.m

    The mathematics is discussed here:
    Brand, M. Fast low-rank modifications of the thin singular value decomposition.
    Linear Algebra and its Applications 415, 20-30 (2006).

    Note that there are some differences between Numpy and Matlab implementations
    of SVD.  So far, I noticed that the V matrix in the X = USV decomposition is
    transposed when comparing Matlab to Numpy.  Also, the ordering of the diagonal
    S matrix is different; Numpy is more economical since only the diagonals are
    specified as a 1D matrix.  More importantly, the actual diagonal entries
    appear to be reversed in their ordering by comparison.  I don't know how they
    differ in the event of degenerate values.  There are likely some "gotcha's"
    that I am not yet aware of.

    Original documenation by D. Wingate 8/17/2007:
    =================================================================
    Given the SVD of

      X = U*S*V'

    update it to be the SVD of

        [X A] = Up*Sp*Vp'

    that is, add new columns (ie, data points).

    I have found that it is faster to add several (say, 200) data points
    at a time, rather than updating it incrementally with individual
    data points (for 500 dimensional vectors, the speedup was roughly
    25x).  However, in the rank-one case there is structure that I have
    not exploited, so that may still be faster than a block method.

    The subspace rotations involved may not preserve orthogonality due
    to numerical round-off errors.  To compensate, you can set the
    "force_orth" flag, which will force orthogonality via a QR plus
    another SVD.  In a long loop, you may want to force orthogonality
    every so often.

    See Matthew Brand, "Fast low-rank modifications of the thin
    singular value decomposition".

    D. Wingate 8/17/2007
    =====================================================================

    :param U: Left singular vectors of shape p x q
    :param S: Diagonal matrix (shape q -- only the q diagonal elements specified)
    :param V: Right singular vectors of shape q x n
    :param A: The matrix to be appended to X = USV (shape p x n)
    :param force_orth: Force orthogonality
    :return:
    '''

    current_rank = U.shape[1]

    # P is an orthogonal basis of the column-space
    # of (I-UU')a, which is the component of "a" that is
    # orthogonal to U.
    m = np.dot(U.T, A)
    p = A - np.dot(U, m)
    # orth documentation: "Construct an orthonormal basis for the range of p using SVD"
    P = orth(p)
    # p may not have full rank.  If not, P will be too small.  Pad with zeros.
    if (p.shape[1] - P.shape[1]) > 0:
        padder =  np.zeros((P.shape[0], p.shape[1]-P.shape[1]))
        if P.shape[1] == 0:
            P = padder
        else:
            P = np.vstack([P, padder])

    # Note, vstack(NxM, NxM) -->  2N x M array.
    Ra = np.dot(P.T, p)
    # Diagonalize K, maintaining rank

    z = np.zeros_like(m)
    K = np.vstack([np.hstack([np.diag(S), m]), np.hstack([z.T, Ra])])
    [tUp, tSp, tVp] = svds(K, current_rank)
    # Now update our matrices!
    Sp = tSp.copy()

    Up = np.dot(np.hstack([U, P]), tUp)  # this may not preserve orthogonality over many repetitions.  See below.

    # Exploit structure to compute this fast: Vp = [ V Q ] * tVp;
    Vp = np.dot(V.T, tVp.T[0:current_rank, :]).T
    Vp = np.vstack([Vp.T, tVp.T[current_rank:tVp.shape[0], :]]).T

    # The above rotations may not preserve orthogonality, so we explicitly
    # deal with that via a QR plus another SVD.  In a long loop, you may
    # want to force orthogonality every so often.
    if force_orth:
        raise NotImplementedError('This has not been tested yet.')
        [UQ, UR] = qr(Up, mode='economic')
        [VQ, VR] = qr(Vp, mode='economic')
        print(Up.shape, Vp.shape, UR.shape, Sp.shape, VR.shape)
        [tUp, tSp, tVp] = svds(np.dot(UR, np.dot(Sp, VR.T)), current_rank)
        Up = np.dot(UQ, tUp)
        Vp = np.dot(VQ, tVp)
        Sp = tSp

    return Up, Sp, Vp
