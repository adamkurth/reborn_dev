from reborn.target import placer
import numpy as np
np.random.seed(10)


def test_01():
    for i in range(3):
        a = placer.particles_in_a_sphere(10, 100, 1)
        assert(np.max(np.sqrt(np.sum(a**2, axis=-1))) < 4.5)
