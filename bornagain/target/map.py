from __future__ import division

import numpy as np

from bornagain.target import crystal


class CrystalMeshTool(object):

    ''' Should be helpful when working with 3D density maps and intensity maps.
    we'll see how the interface evolves...'''

    sym_luts = None
    n_vecs = None
    x_vecs = None
    r_vecs = None
    h_vecs = None

    def __init__(self, cryst, resolution, oversampling):

        ''' This should intelligently pick the limits of a map.  Documentation to follow later...
        '''

        d = resolution
        s = np.ceil(oversampling)

        abc = np.array([cryst.a, cryst.b, cryst.c])

        m = 1
        for T in cryst.symTs:
            for mp in np.arange(1, 10):
                Tp = T*mp
                if np.round(np.max(Tp % 1.0)*100)/100 == 0:
                    if mp > m:
                        m = mp
                    break

        Nc = np.max(np.ceil(abc / (d * m)) * m)

        self.cryst = cryst
        self.m = np.int(m)
        self.s = np.int(s)
        self.d = abc / Nc
        self.dx = 1 / Nc
        self.Nc = np.int(Nc)
        self.N = np.int(Nc * s)
        self.P = np.int(self.N**3)
        self.w = np.array([self.N**2, self.N, 1])

    def get_n_vecs(self):

        P = self.P
        N = self.N
        p = np.arange(0, P)
        n_vecs = np.zeros([P, 3])
        n_vecs[:, 0] = np.floor(p / (N**2))
        n_vecs[:, 1] = np.floor(p / N) % N
        n_vecs[:, 2] = p % N
        # n_vecs = n_vecs[:, ::-1]
        return n_vecs

    def get_x_vecs(self):

        x_vecs = self.get_n_vecs()
        x_vecs = x_vecs * self.dx
        return x_vecs

    def get_r_vecs(self):

        x = self.get_x_vecs()

        return np.dot(self.cryst.O, x.T).T

    def get_h_vecs(self):

        h = self.get_n_vecs()
        h = h / self.dx / self.N
        f = np.where(h.ravel() > (self.Nc/2))
        h.flat[f] = h.flat[f] - self.Nc
        return h

    def get_sym_lut(self, i):

        return self.get_sym_luts()[i]

    def get_sym_luts(self):

        if self.sym_luts is None:

            sym_luts = []
            x0 = self.get_x_vecs()

            for (R, T) in zip(self.cryst.symRs, self.cryst.symTs):
                lut = np.dot(R, x0.T).T + T  # transform x vectors in 3D grid
                lut /= self.dx               # switch from x to n vectors
                lut = lut % self.N           # wrap around
                lut = np.dot(self.w, lut.T)  # in p space
                sym_luts.append(lut.astype(np.int))

            self.sym_luts = sym_luts

        return self.sym_luts

    def symmetry_transform(self, i, j, data):

        luts = self.get_sym_luts()
        data_trans = np.zeros(data.shape)
        data_trans.flat[luts[j]] = data.flat[luts[i]]

        return data_trans

    def reshape(self, data):

        N = self.N
        return data.reshape([N, N, N])

    def reshape3(self, data):

        N = self.N
        return data.reshape([N, N, N, 3])

    def zeros(self):

        N = self.N
        return np.zeros([N, N, N])


if __name__ == '__main__':

    import numpy as np

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Crystal information
    sg = 'P 1'
    a = 281.000e-10
    b = 281.000e-10
    c = 165.200e-10
    alpha = 90.0 * np.pi / 180.0
    beta  = 90.0 * np.pi / 180.0
    gamma = 120.0 * np.pi / 180.0
    cryst = crystal.structure()
    cryst.set_cell(a, b, c, alpha, beta, gamma)
    cryst.set_spacegroup(sg)

    # Desired properties of the map
    d = 9e-9  # Minimum resolution
    s = 2        # Oversampling factor

    # Create a meshtool
    mt = CrystalMeshTool(cryst, d, s)

    print(mt.N)

    n = mt.get_n_vecs()
    x = mt.get_x_vecs()
    r = mt.get_r_vecs()
    h = mt.get_h_vecs()
    # print(h)

    # Check for interpolation artifacts
    # for (R, T) in zip(cryst.symRs, cryst.symTs):
    #     print(R)
    #     print(T)
    #     xp = np.dot(R, x.T).T
    #     xp = xp + T
    #     xp = xp/mt.dx
    #     xpp = xp - np.round(xp)
    #     print(np.max(np.abs(xpp)))

    v = h
    v3 = mt.reshape3(v)

    # sym_luts = mt.get_sym_luts()
    # for lut in sym_luts:
    #     print(lut)

    # Display the mesh as a scatterplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c='k', marker='.', s=1)
    ax.scatter(v3[0, 0, :, 0], v3[0, 0, :, 1], v3[0, 0, :, 2], c='r', marker='.', s=200, edgecolor='')
    ax.scatter(v3[0, :, 0, 0], v3[0, :, 0, 1], v3[0, :, 0, 2], c='g', marker='.', s=200, edgecolor='')
    ax.scatter(v3[:, 0, 0, 0], v3[:, 0, 0, 1], v3[:, 0, 0, 2], c='b', marker='.', s=200, edgecolor='')
    ax.set_aspect('equal')
    plt.show()

    print('done!')