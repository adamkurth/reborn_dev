import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def meshplot3D(f, isosurface_value):
    # Use marching cubes to obtain a surface mesh
    if isosurface_value < np.min(f):
        isosurface_value = np.min(f)*1.1
    verts, faces = measure.marching_cubes(f, isosurface_value)

    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: "verts[faces]" to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(0, f.shape[0])
    ax.set_ylim(0, f.shape[1])
    ax.set_zlim(0, f.shape[2])

    plt.show()