from pyopencl import array as clarray
from reborn.simulate.clcore import ClCore, help, print_device_info
cl_array = clarray.Array
havecl = True
print('In test.  Just one line after this one!')
help()
test_core = ClCore(context=None, queue=None, group_size=1, double_precision=True, debug=1)
# if test_core.double_precision:
#     have_double = True
# else:
#     have_double = False
# ctx = reborn.simulate.clcore.create_some_gpu_context()
# except ImportError:
#     ClCore = None
#     clarray = None
#     pyopencl = None
#     havecl = False


# def func(vecs):
#     return vecs[:, 0].ravel().copy()


# def func1(vecs):
#     return np.sin(vecs[:, 0]/10.0) + np.cos(3*vecs[:, 1]/10.) + np.cos(2*vecs[:, 2]/10.)
#
#
# def test_interpolations_01():
#
#     # if ClCore is None:
#     #     return
#
#     clcore = ClCore(group_size=32)
#     shape = (6, 7, 8)
#     dens = np.ones(shape, dtype=clcore.real_t)
#     corners = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vectors = np.ones((shape[0], 3), dtype=clcore.real_t)
#     vectors[:, 0] = np.arange(0, shape[0]).astype(clcore.real_t)
#     dens_gpu = clcore.to_device(dens)
#     vectors_gpu = clcore.to_device(vectors)
#     dens2 = clcore.mesh_interpolation(dens_gpu, vectors_gpu, q_min=corners, dq=deltas, N=shape)
#     assert np.max(np.abs(dens2)) > 0
#     assert np.max(np.abs(dens2[:] - 1)) < 1e-6
#
#
# def test_interpolations_02():
#
#     # if ClCore is None:
#     #     return
#
#     clcore = ClCore(group_size=32)
#     shape = (6, 7, 8)
#     corners = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     x, y, z = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), np.arange(0, shape[2]), indexing='ij')
#     vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t)
#     dens = func(vectors0).reshape(shape)
#     x, y, z = np.meshgrid(np.arange(1, shape[0]-2), np.arange(1, shape[1]-2), np.arange(1, shape[2]-2), indexing='ij')
#     vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t) + 0.1
#     dens1 = func(vectors)
#     dens2 = np.zeros_like(dens1)
#     dens2_gpu = clcore.to_device(dens2)
#     clcore.mesh_interpolation(dens, vectors, q_min=corners, dq=deltas, N=shape, a=dens2_gpu)
#     dens2 = dens2_gpu.get()
#     assert np.max(np.abs(dens1)) > 0
#     assert np.max(np.abs(dens2)) > 0
#     assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-6


# def test_interpolations_03():
#
#     # if ClCore is None:
#     #     return
#
#     clcore = ClCore(group_size=32)
#     nx, ny, nz = 6, 7, 8
#     corners = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
#     vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t)
#     dens = func1(vectors0).reshape([nx, ny, nz])
#     x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
#     vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t) + 0.1
#     dens1 = func1(vectors)
#     dens2 = np.zeros_like(dens1)
#     dens2_gpu = clcore.to_device(dens2, dtype=clcore.real_t)
#     clcore.mesh_interpolation(dens, vectors, q_min=corners, dq=deltas, N=dens.shape, a=dens2_gpu)
#     dens2 = dens2_gpu.get()
#     assert np.max(np.abs(dens1)) > 0
#     assert np.max(np.abs(dens2)) > 0
#     assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-2


# def test_insertions_01():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = np.array([[2, 3, 4], [3, 4, 4], [4, 4, 4]], dtype=clcore.real_t)
#     vals = func1(vecs).astype(clcore.real_t)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, deltas=deltas, corner=corner)
#     densities = densities_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert np.max(np.abs((vals[0] - densities[2, 3, 4]) / vals)) < 1e-8


# def test_insertions_02():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = np.array([[2, 3, 4], [2.1, 3.1, 4.1], [1.9, 2.9, 3.9]], dtype=clcore.real_t)
#     vals = func1(vecs)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, corner=corner, deltas=deltas)
#     densities = densities_gpu.get()
#     weights = weights_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert np.max(np.abs((vals[0] - densities[2, 3, 4]/weights[2, 3, 4]) / vals[0])) < 1e-3


# def test_insertions_03():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     np.random.seed(0)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = np.random.rand(1000, 3) * (np.array(shape)-1).astype(clcore.real_t)
#     vecs = np.floor(vecs)
#     vals = func(vecs)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, corner=corner, deltas=deltas)
#     val = func(np.array([[2, 3, 4]], dtype=clcore.real_t))
#     densities = densities_gpu.get()
#     weights = weights_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((val - densities[2, 3, 4]/weights[2, 3, 4]) / val)) < 1e-8


# def test_insertions_04():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     np.random.seed(0)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = (np.random.rand(10000, 3) * (np.array(shape)-1)).astype(clcore.real_t)
#     vals = func1(vecs)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, corner=corner, deltas=deltas)
#     val = func1(np.array([[2, 3, 4]], dtype=clcore.real_t))
#     densities = densities_gpu.get()
#     weights = weights_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((val - densities[2, 3, 4]/weights[2, 3, 4]) / val)) < 1e-2
