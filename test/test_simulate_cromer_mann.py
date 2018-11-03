import numpy as np
from bornagain.simulate import clcore


def test_no_atomic_form_factor():
    
    core = clcore.ClCore(double_precision=False)

    Npix = 512*512
    Natom = 1000

    q = np.random.random((Npix, 3))
    r = np.random.random((Natom, 3))
    f = np.ones(Natom, dtype=np.complex64)
    A = core.phase_factor_qrf(q, r, f)

    # print (A[0])
#   now test the cromermann simulation
#     print("Testing cromermann")
    core.init_amps(Npix)
    core.prime_cromermann_simulator(q, None)
    q_cm = core.get_q_cromermann()
    r_cm = core.get_r_cromermann(r, sub_com=False)
    core.run_cromermann(q_cm, r_cm, rand_rot=False)
    A2 = core.release_amplitudes()

    assert(np.allclose(A, A2))


if __name__ == "__main__":

    test_no_atomic_form_factor()
