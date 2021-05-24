import numpy as np
from reborn.simulate.examples import lysozyme_pdb_file, psi_pdb_file
from reborn.target import crystal, density


def test_01():

    cryst = crystal.CrystalStructure(psi_pdb_file)
    dens = crystal.CrystalDensityMap(cryst, 20e-10, 2)
    assert np.sum(np.abs(dens.n_vecs[1, :] - np.array([0, 0, 1]))) < 1e-8
    assert np.allclose(dens.x_vecs[1, :], np.array([0, 0., 0.1]))

    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    for d in np.array([5, 10])*1e-10:

        dens = crystal.CrystalDensityMap(cryst, d, 1)
        dat0 = np.reshape(np.arange(0, dens.size), dens.shape).astype(float)
        dat1 = dens.symmetry_transform(0, 1, dat0)
        dat2 = dens.symmetry_transform(1, 0, dat1)

        assert np.allclose(dat0, dat2)

    cryst = crystal.CrystalStructure(psi_pdb_file)
    for d in np.array([5, 10])*1e-10:

        dens = crystal.CrystalDensityMap(cryst, d, 2)
        dat0 = np.reshape(np.arange(0, dens.size), dens.shape).astype(float)
        dat1 = dens.symmetry_transform(0, 1, dat0)
        dat2 = dens.symmetry_transform(1, 0, dat1)

        assert np.allclose(dat0, dat2)


def test_02():

    cryst = crystal.CrystalStructure(psi_pdb_file)

    for d in [0.2, 0.3, 0.4, 0.5]:

        mt = crystal.CrystalDensityMap(cryst, d, 1)
        dat0 = np.reshape(np.arange(0, mt.size), mt.shape).astype(float)
        dat1 = mt.symmetry_transform(0, 1, dat0)
        dat2 = mt.symmetry_transform(1, 0, dat1)

        assert np.allclose(dat0, dat2)


def func(vecs):
    return vecs[:, 0].ravel().copy()


def func1(vecs):
    return np.sin(vecs[:, 0]/10.0) + np.cos(3*vecs[:, 1]/10.) + np.cos(2*vecs[:, 2]/10.)


def test_03():

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    dens = np.ones([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = np.ones((nx, 3), dtype=float_t)
    vectors[:, 0] = np.arange(0, nx).astype(float_t)
    dens2 = density.trilinear_interpolation(dens, vectors, corners, deltas)
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs(dens2[:] - 1)) < 1e-6

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t) + 0.1
    dens1 = func(vectors)
    dens2 = np.zeros_like(dens1)
    density.trilinear_interpolation(dens, vectors, corners, deltas, out=dens2)
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-8

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func1(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t) + 0.1
    dens1 = func1(vectors)
    dens2 = np.zeros_like(dens1)
    density.trilinear_interpolation(dens, vectors, corners, deltas, out=dens2)
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-2

    # Check that the above works with complex density map
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func1(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t) + 0.1
    dens1 = func1(vectors).astype(np.complex128)
    dens = dens.astype(np.complex128)
    dens += dens*2j
    dens1 += dens1*2j
    dens2 = np.zeros_like(dens1)
    density.trilinear_interpolation(dens, vectors, corners, deltas, out=dens2)
    assert np.max(np.abs(np.real(dens1))) > 0
    assert np.max(np.abs(np.real(dens2))) > 0
    assert np.max(np.abs((np.real(dens1) - np.real(dens2))/np.real(dens1))) < 1e-2
    assert np.max(np.abs(np.imag(dens1))) > 0
    assert np.max(np.abs(np.imag(dens2))) > 0
    assert np.max(np.abs((np.imag(dens1) - np.imag(dens2))/np.imag(dens1))) < 1e-2


def test_04():

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = np.array([[2, 3, 4]], dtype=float_t)
    vals = func1(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((vals - densities[2, 3, 4]) / vals)) < 1e-8

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = np.array([[2.5, 3.5, 4.5]], dtype=float_t)
    vals = func1(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((vals - densities[2, 3, 4]/counts[2, 3, 4]) / vals)) < 1e-8

    np.random.seed(0)
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(float_t)
    vectors = np.floor(vectors)
    vals = func(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    val = func(np.array([[2, 3, 4]], dtype=float_t))
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-8

    np.random.seed(0)
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(float_t)
    vals = func1(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    val = func1(np.array([[2, 3, 4]], dtype=float_t))
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-2

    np.random.seed(0)
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=np.complex128)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(float_t)
    vals = func1(vectors)
    vals = vals + 2j*vals
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    val = func1(np.array([[2, 3, 4]], dtype=float_t))
    val = val + 2j*val
    assert np.max(np.abs(np.real(densities))) > 0
    assert (np.abs((np.real(val) - np.real(densities[2, 3, 4])/counts[2, 3, 4]) / np.real(val))) < 1e-2
    assert np.max(np.abs(np.imag(densities))) > 0
    assert (np.abs((np.imag(val) - np.imag(densities[2, 3, 4])/counts[2, 3, 4]) / np.imag(val))) < 1e-2

def test_05():

    # Check that atom scattering factors are additive

    x_min = np.array([-5, -10, -12], dtype=np.double)
    x_max = np.array([+5, +10, +13], dtype=np.double)
    shape = np.array([11, 21, 26], dtype=np.double)
    x_vecs = np.array([[5.5, 10.5, 13.5], [5.5, 10.5, 13.5]], dtype=np.double)
    sigma = 1.0
    f = np.array([1, 1], dtype=np.double)
    orth_mat = np.eye(3, dtype=np.double)
    sum_map = density.build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat)
    assert sum_map[0, 0, 0] == sum_map[-1, -1, -1]
    assert np.abs(np.sum(sum_map) - f[0]*2) / np.abs(f[0]*2) < 1e-8


def test_06():

    # Check that wrap-around (periodic condition) is satisfied and max radius truncation

    x_min = np.array([-5, -10, -12], dtype=np.double)
    x_max = np.array([+5, +11, +13], dtype=np.double)
    shape = np.array([11, 22, 26], dtype=np.double)
    x_vecs = np.array([[5.5, 11.5, 13.5]], dtype=np.double)
    sigma = 1.0
    f = np.array([1], dtype=np.double)
    orth_mat = np.eye(3, dtype=np.double)
    sum_map = density.build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, max_radius=4)
    assert sum_map[0, 0, 0] == sum_map[-1, -1, -1]
    assert np.abs(np.sum(sum_map) - f[0]) / np.abs(f[0]) < 1e-8
    w = np.where(sum_map > 0)
    assert len(w[0]) == 9**3


def test_07():

    # Check truncation bounds

    x_min = np.array([-5, -10, -12], dtype=np.double)
    x_max = np.array([+5, +11, +13], dtype=np.double)
    shape = np.array([11, 22, 26], dtype=np.double)
    x_vecs = np.array([[-5, -10, -12]], dtype=np.double)
    sigma = 1.0
    f = np.array([1], dtype=np.double)
    orth_mat = np.eye(3, dtype=np.double)
    sum_map = density.build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, max_radius=1)
    assert sum_map[2, 0, 0] == 0
    assert sum_map[0, 2, 0] == 0
    assert sum_map[0, 0, 2] == 0
    norm = 1 + 6*np.exp(-1/(2*sigma**2)) + 12*np.exp(-1/sigma**2) + 8*np.exp(-np.sqrt(3)**2/(2*sigma**2))
    assert np.abs(sum_map[1, 1, 0] - np.exp(-1/sigma**2)/norm) < 1e6
    assert np.abs(sum_map[-1, -1, -1] - np.exp(-np.sqrt(3)**2/(2*sigma**2))/norm) < 1e-6


def test_08():

    # Check orthogonalization matrix

    x_min = np.array([-5, -10, -12], dtype=np.double)
    x_max = np.array([+5, +11, +13], dtype=np.double)
    shape = np.array([11, 22, 26], dtype=np.double)
    x_vecs = np.array([[-5, -10, -12]], dtype=np.double)
    sigma = 1.0
    orth_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]])  # Columns are crystal basis vectors => shrinks "c" axis.
    f = np.array([1], dtype=np.double)
    sum_map = density.build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat=orth_mat,
                                                          max_radius=1)
    assert sum_map[2, 0, 0] == 0
    assert sum_map[0, 2, 0] == 0
    assert sum_map[0, 0, 2] != 0
    assert sum_map[0, 0, -2] != 0
