
import numpy as np
from scipy.spatial import distance, cKDTree


class Place(cKDTree):
    def __init__(self, box_edge, min_dist, max_try=10000, *args, **kwargs):
        """
        Place points into a box of edge length box_edge, and don't let any two points
        
        Parameters
        ==========
        get with    t0 = time()in min_dist from one another.

        box_edge, float
            side length of the box to place spheres into
        min_dist, float
            minimum distance between two points in the box
        max_try, int
            number of times to try placing a new point such 
            that is does not overlap
        
        """
        np.random.seed()
        a = np.random.uniform(0, box_edge, (1, 3))
        cKDTree.__init__(self, a, *args, **kwargs)
        self.min_dist = min_dist
        self.box_edge = box_edge
        self.max_try = max_try
        self.too_dense = False

    def insert(self):
        """adds a new point to the box"""
        new_pt = np.random.uniform(0, self.box_edge, (1, 3))
        n_try = 0
        # @dermen - why is the inequality comparing with inf?
        is_overlapping = self.query(new_pt, distance_upper_bound=self.min_dist)[
                             0] < np.inf  # query for a nearest neighbor
        while is_overlapping:
            new_pt = np.random.uniform(0, self.box_edge, (1, 3))
            is_overlapping = self.query(new_pt, distance_upper_bound=self.min_dist)[0] < np.inf
            n_try += 1
            if n_try > self.max_try:
                print("Getting too tight in here!")
                self.too_dense = True
                return
        data = np.concatenate((self.data, new_pt))  # combine new pt and old pts
        super(Place, self).__init__(data)  # re-initialize the parent class with new data


def place_spheres(Vf, sph_rad=1., box_edge=None, Nspheres=1000, tol=0.01):
    """
    Vf, float
        Fraction of sample volume occupied by spheres
    Nspheres, int
        how many spheres in the sample volume
    tol, float
        minimum distance the unit spheres can be to one another
    """
    #   volume of a unit sphere
    sph_vol = (4 / 3.) * np.pi * (sph_rad) ** 3

    if box_edge is not None:
        #       then we let Nspheres be a free
        box_vol = box_edge ** 3
        Nspheres = int((box_vol * Vf) / sph_vol)
    else:
        #       then Nspheres determines the size of the box
        box_vol = sph_vol * Nspheres / Vf
        box_edge = np.power(box_vol, 1 / 3.)

    min_dist = 2 * sph_rad + tol  # diameter plus tol,

    print("Placing %d spheres into a box of side length %.4f" % (Nspheres, box_edge))

    p = Place(box_edge, min_dist)  # init the Placer
    while p.n < Nspheres:
        p.insert()  # insert pt!
        if p.too_dense:
            print("\tbreaking insert loop with %d/%d spheres" % (p.n, Nspheres))
            break

    return p.data




