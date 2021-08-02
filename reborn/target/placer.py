r"""
Utilities for positioning objects (x-ray targets) in space.
"""

import numpy as np
from scipy.spatial import distance, cKDTree


class Place(cKDTree):

    def __init__(self, box_edge, min_dist, max_try=10000, *args, **kwargs):
        r"""
        Utility for placing points into a box while ensuring that no two points get closer than some minimum distance.
        This is based on the scipy.spatial.cKDTree class.

        Note:
          * Since an iterative process is employed, make sure that you use reasonable values for the minimum distance
            and box size.  It is of course possible to choose parameters for which it is impossible to place the
            spheres.
          * Since this is a sub-class of scipy.spatial.cKDTree, you may pass arguments that are relevant to that class.

        Contributed by Derek Mendez.

        Arguments:
            box_edge (float): Side length of the box to place spheres into.
            min_dist (float): Minimum distance between two points in the box.
            max_try (int): Number of times to try placing a new point such that is does not overlap.
        """
        a = np.random.uniform(0, box_edge, (1, 3))
        cKDTree.__init__(self, a, *args, **kwargs)
        self.min_dist = min_dist
        self.box_edge = box_edge
        self.max_try = max_try
        self.too_dense = False

    def insert(self):
        """Adds one new point to the box."""
        new_pt = np.random.uniform(0, self.box_edge, (1, 3))
        n_try = 0
        # @dermen - why is the inequality comparing with inf?
        is_overlapping = self.query(new_pt, distance_upper_bound=self.min_dist)[
            0] < np.inf  # query for a nearest neighbor
        while is_overlapping:
            new_pt = np.random.uniform(0, self.box_edge, (1, 3))
            is_overlapping = self.query(
                new_pt, distance_upper_bound=self.min_dist)[0] < np.inf
            n_try += 1
            if n_try > self.max_try:
                print("Getting too tight in here!")
                self.too_dense = True
                return
        # combine new pt and old pts
        data = np.concatenate((self.data, new_pt))
        # re-initialize the parent class with new data
        super(Place, self).__init__(data)


def place_spheres(volume_fraction, radius=1., box_edge=None, n_spheres=1000, tol=0.01):
    """
    Contributed by Derek Mendez.

    No documentation.

    Arguments:
        volume_fraction (float): Fraction of sample volume occupied by spheres.
        radius (float): Undocumented.
        box_edge (float): Undocumented.
        n_spheres (int): How many spheres in the sample volume.
        tol (float): Minimum distance the unit spheres can be to one another.

    Returns: Ask Derek.
    """
    #   volume of a unit sphere
    sph_vol = (4 / 3.) * np.pi * radius ** 3

    if box_edge is not None:
        #       then we let Nspheres be a free
        box_vol = box_edge ** 3
        n_spheres = int((box_vol * volume_fraction) / sph_vol)
    else:
        #       then Nspheres determines the size of the box
        box_vol = sph_vol * n_spheres / volume_fraction
        box_edge = np.power(box_vol, 1 / 3.)

    min_dist = 2 * radius + tol  # diameter plus tol,

    print("Placing %d spheres into a box of side length %.4f" % (n_spheres, box_edge))

    p = Place(box_edge, min_dist)  # init the Placer
    while p.n < n_spheres:
        p.insert()  # insert pt!
        if p.too_dense:
            print("\tbreaking insert loop with %d/%d spheres" % (p.n, n_spheres))
            break

    return p.data


def particles_in_a_sphere(sphere_diameter, n_particles, particle_diameter, max_attempts=1e6):
    r"""
    Place particles randomly in a spherical volume.  Assume particles are spheres and they cannot touch any other
    particle.  Also assumes that surface of spherical particles cannot extend beyond the surface of the containing
    sphere.

    Args:
        sphere_radius:
        n_particles:
        particle_diameter:
        max_attempts:

    Returns:

    """
    rmax2 = ((sphere_diameter - particle_diameter)/2)**2  # Note: the particle cannot extend outside of the sphere
    sqrtpd = np.sqrt(particle_diameter)
    if particle_diameter > sphere_diameter:
        raise ValueError("Particle diameter is larger than sphere diameter.")
    pos_vecs = np.zeros((n_particles, 3))
    for i in range(n_particles):
        for a in range(int(max_attempts)):
            vec = (np.random.rand(3) - 0.5)*rmax2  # Random position ranging from -r to +r
            vmag = np.sum(vec**2)
            if vmag > rmax2:  # Check if it's in the sphere
                continue
            if i > 0:  # No neighbors for the first particle
                mindist = np.min(np.sum((pos_vecs[0:i] - vec)**2))  # Check closest neighbor
                if mindist < sqrtpd:
                    continue
            break  # If we made it here, success!
        pos_vecs[i, :] = vec
        if a == int(max_attempts) - 1:
            print('Failed to place all particles!!')
    return pos_vecs







