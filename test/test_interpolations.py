from reborn import utils
from reborn.target import density
import numpy as np

# The usage of N_data and N_pattern in some of the tests below can be interpreted as:
# N_data    : Number of pixels in a detector
# N_pattern : Number of patterns measured from that detector
# And the action of the trilinear_insert() is to insert the values measured by the detector
# into a 3D regularly grided array.


def test_1():
    data_coord = np.array([[0.1, 0, 0]], dtype=np.float64)
    data_val = np.array([1], dtype=np.float64)
    mask = np.ones(len(data_val), dtype=np.float64)

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0. ],
                    [ 0. ,  0. ,  0. ],
                    [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                    [ 0. ,  1.0,  0. ],
                    [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                    [ 0. ,  1.0,  0. ],
                    [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")


    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9

    dataout = np.zeros(N_bin, dtype=np.float64)
    weightout = np.zeros(N_bin, dtype=np.float64)
    density.trilinear_insertion(dataout, weightout, data_coord, data_val, x_min=x_min, x_max=x_max)
    w = np.where(weightout != 0)
    data_avg = dataout.copy()
    data_avg[w] /= weightout[w]
    assert np.sum(np.abs(data_avg - ans)) < 1e-9





def test_2():
    data_coord = np.array([[0.5, 0.5, 0.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1.0,  1.0],
                    [ 0.   ,  1.0,  1.0]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1.0,  1.0],
                    [ 0.   ,  1.0,  1.0]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")


    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_3(): # Boundary test
    data_coord = np.array([[-1.0, -1.0, -1.0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 1.,  0.,  0.],
                    [ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]],

                   [[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]],

                   [[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_4(): # Boundary test
    data_coord = np.array([[1.0, 1.0, 1.0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],

           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],

           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  1.]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_5():
    data_coord = np.array([[-0.5, -0.5, -0.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 1.0,  1.0,  0.   ],
                    [ 1.0,  1.0,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]],

                   [[ 1.0,  1.0,  0.   ],
                    [ 1.0,  1.0,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_6():
    data_coord = np.array([[0, 0, 0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]],

           [[ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  0.]],

           [[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_7(): # Boundary padding test
    data_coord = np.array([[-1.4999, -1.4999, -1.4999]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 1.0,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]]])


    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-6


def test_8(): # Boundary padding test
    data_coord = np.array([[1.4999, 1.4999, 1.4999]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-6


def test_9():
    data_coord = np.array([[-1.0, 1.0, 1.0]])
    data_val = np.array([20.7])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[  0. ,   0. ,   0. ],
            [  0. ,   0. ,   0. ],
            [  0. ,   0. ,  20.7]],

           [[  0. ,   0. ,   0. ],
            [  0. ,   0. ,   0. ],
            [  0. ,   0. ,   0. ]],

           [[  0. ,   0. ,   0. ],
            [  0. ,   0. ,   0. ],
            [  0. ,   0. ,   0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_10():
    data_coord = np.array([[-1.3, 0.2, 1.48]])
    data_val = np.array([213.7])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,  213.7],
            [  0.     ,   0.     ,  213.7]],

           [[  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ]],

           [[  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_11():
    N_data = 10
    data_coord = np.array([[ 0.72128486,  0.22625585,  0.86686662],
           [ 0.25203022,  0.49918326,  0.38627506],
           [ 0.62419278,  0.4668094 ,  0.27941368],
           [ 0.86714749,  0.52950898,  0.52951549],
           [ 0.9085269 ,  0.85187713,  0.14500404],
           [ 0.63341219,  0.68095018,  0.45570456],
           [ 0.06567521,  0.16985335,  0.81385588],
           [ 0.91764472,  0.14515303,  0.0895495 ],
           [ 0.87531152,  0.73717891,  0.23602149],
           [ 0.60331596,  0.9362185 ,  0.05913338]])
    data_val = np.ones(N_data)
    mask = np.ones(N_data)

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  1.0,  1.0],
            [ 0.        ,  1.0,  1.0]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  1.0,  1.0],
            [ 0.        ,  1.0,  1.0]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-7


def test_12():
    N_data = 2
    data_coord = np.array([[1.0, 1.0, 1.0], [0,0,0]])
    data_val = np.array([1, 1])
    mask = np.ones(N_data)

    N_bin = np.array([4, 3, 4])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  1.0,   1.0,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  1.0 ,  1.0,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  1.  ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_13():
    N_data = 2
    data_coord = np.array([[1.0, 1.0, 1.0], [0,0,0]])
    data_val = np.array([1, 10])
    mask = np.ones(N_data)

    N_bin = np.array([3, 5, 4])
    x_min = np.array([-3, -1, -6])
    x_max = np.array([2, 3, 2])

    ans = np.array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  10   ,  10   ],
            [ 0.   ,  0.   ,  1 ,  1 ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  10  ,  10  ],
            [ 0.   ,  0.   ,  1,  1],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_14(): # Out of bounds test
    N_data = 2
    data_coord = np.array([[12.0, 13.0, 14.0], [-10,-10,-20]])
    data_val = np.array([1, 10])
    mask = np.ones(N_data)

    N_bin = np.array([3, 5, 4])
    x_min = np.array([-3, -1, -6])
    x_max = np.array([2, 3, 2])

    ans = np.array([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_15(): # More-than-one-patterns test
    N_data = 2
    N_pattern = 3

    data_coord = np.array([[1.0, 1.0, 1.0], [0,0,0]])
    data_coord = np.tile(data_coord, (N_pattern,1))
    data_val = np.array([1, 10, 1, 10, 1, 10])
    mask = np.ones(N_data * N_pattern)

    N_bin = np.array([3, 5, 4])
    x_min = np.array([-3, -1, -6])
    x_max = np.array([2, 3, 2])

    ans = np.array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  10   ,  10   ],
            [ 0.   ,  0.   ,  1 ,  1 ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  10   ,  10  ],
            [ 0.   ,  0.   ,  1, 1],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_16():
    N_data = 7

    data_coord = np.array([[1.0, 1.0, 1.0], \
                           [0,0,0], \
                           [-56.912701, -89.495757,  -69.0889544], \
                           [7.5068156,  74.737885,  -16.23046],  \
                           [400.71784,  60.451844,  -33.82541], \
                           [-23.791852,  70.368846,  0.36841697], \
                           [-90.728055,  0.40624265,  -0.82445448]])
    data_val = np.array([ 19.43855069,  0.17395744,  1.306821901,  6.04473423, \
            22.1702885,   5.80547311,  152.1111933])
    mask = np.ones(N_data)

    N_bin = np.array([5, 4, 6])
    x_min = np.array([-100, -99, -71])
    x_max = np.array([200, 300, -2])

    ans = np.array([[[ 1.3068219,  1.3068219,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 1.3068219,  1.3068219,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]],

       [[ 1.3068219,  1.3068219,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 1.3068219,  1.3068219,  0.        ,  6.04473423,  6.04473423,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  6.04473423,  6.04473423,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]],

       [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  6.04473423,  6.04473423,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  6.04473423,  6.04473423 ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]],

       [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]],

       [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-7


def test_17(): # Mask test
    N_data = 2
    data_coord = np.array([[1.0, 1.0, 1.0], [0,0,0]])
    data_val = np.array([1, 1])
    mask = np.array([0,1])

    N_bin = np.array([4, 3, 4])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  1.0,  1.0,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  1.0,  1.0,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")
    
    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9

    # Check that the shape of the output is equal to N_bin and the shape of the answer
    assert np.sum(N_bin - np.array(dataout.shape)) == 0
    assert np.sum(ans.shape - np.array(dataout.shape)) == 0


def test_18(): # Out of bounds test2
    N_data = 2
    data_coord = np.array([[0.2, 0.7, 14.0], [-0.1,-16,-63]])
    data_val = np.array([1, 10])
    mask = np.ones(N_data)

    N_bin = np.array([3, 5, 4])
    x_min = np.array([-3, -1, -6])
    x_max = np.array([2, 3, 2])

    ans = np.array([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    

    assert np.sum(np.abs(dataout - ans)) < 1e-9

    # Check that the shape of the output is equal to N_bin and the shape of the answer
    assert np.sum(N_bin - np.array(dataout.shape)) == 0
    assert np.sum(ans.shape - np.array(dataout.shape)) == 0


def test_19(): # Complex value test 1 - single complex value on the exact grid
    data_coord = np.array([[-1.0, -1.0, -1.0]])
    data_val = np.array([10.3+2.9j])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 10.3+2.9j,  0.,  0.],
                     [ 0.       ,  0.,  0.],
                     [ 0.       ,  0.,  0.]],

                    [[ 0.       ,  0.,  0.],
                     [ 0.       ,  0.,  0.],
                     [ 0.       ,  0.,  0.]],

                    [[ 0.       ,  0.,  0.],
                     [ 0.       ,  0.,  0.],
                     [ 0.       ,  0.,  0.]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_20(): # Complex value test 2 - single complex value with interpolations
    data_coord = np.array([[0.5, 0.5, 0.5]])
    data_val = np.array([1j])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1j,  1j],
                    [ 0.   ,  1j,  1j]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1j,  1j],
                    [ 0.   ,  1j,  1j]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_21(): # Complex value test 3 - multiple complex values with interpolations
    data_coord = np.array([[0.1, 0, 0], [0.5, 0.5, 0.5]])
    data_val = np.array([1, 1j])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.87804878+0.12195122j,  1j],
                    [ 0.   ,  1j,  1j]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.44444444+0.55555556j,  1j],
                    [ 0.   ,  1j,  1j]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-7


def test_22(): # Wrap-around test 1 - no wrap-around
    data_coord = np.array([[0.1, 0, 0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  1.0,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  1.0,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_23(): # Wrap-around test 2 - exact wrap-around
    data_coord = np.array([[2, 0, 0]])
    data_val = np.array([99])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0. ],
                     [ 0. ,  99.0,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_24(): # Wrap-around test 3 - wrap-around with interpolation
    data_coord = np.array([[2.1, 0, 0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0. ],
                     [ 0. ,  1.0 ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  1.0,  0. ],
                     [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_25(): # Wrap-around test 4 - wrap-around with interpolation on the boundary
    data_coord = np.array([[1.1, 0, 0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0. ],
                     [ 0. ,  1 ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.,  0. ],
                     [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                     [ 0. ,  1,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_26(): # Wrap-around test 5 - wrap-around with interpolation on the boundary
    data_coord = np.array([[-1.9, 0, 0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0. ],
                     [ 0. ,  1 ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.,  0. ],
                     [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                     [ 0. ,  1,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_27(): # Wrap-around test 6 - wrap-around with interpolation on the boundary
    data_coord = np.array([[-1.9, 0, 0]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 5])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0. ,  0. ,  0.,  0. ,  0. ],
                     [ 0. ,  0. ,  1,  0. ,  0. ],
                     [ 0. ,  0. ,  0.,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0.,  0. ,  0. ],
                     [ 0. ,  0.,  0.,  0. ,  0. ],
                     [ 0. ,  0. ,  0.,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0.,  0. ,  0. ],
                     [ 0. ,  0.,  1,  0. ,  0. ],
                     [ 0. ,  0. ,  0.,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_28(): # Wrap-around test 7 - wrap-around with interpolation on the boundary (corner case)
    data_coord = np.array([[1.5, 1.5, 1.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 1,  0.   ,  1],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 1,  0.   ,  1]],

                    [[ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ]],

                    [[ 1,  0.   ,  1],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 1,  0.   ,  1]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_29(): # Wrap-around test 8 - no wrap-around with interpolation on the boundary (corner case)
    data_coord = np.array([[0.5, 0.5, 0.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.   ,  0.   ]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1.0,  1.0],
                    [ 0.   ,  1.0,  1.0]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1.0,  1.0],
                    [ 0.   ,  1.0,  1.0]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_30(): # Wrap-around test 9 - wrap-around with interpolation on the boundary, non cubic volume
    data_coord = np.array([[1.5, 1.5, 1.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 5])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 1.0,  0.   ,  0.,  0., 0. ],
                     [ 0.   ,  0.   ,  0.,  0., 0.],
                     [ 1.0,  0.   ,  0.,  0., 0. ]],

                    [[ 0.   ,  0.   ,  0.,  0., 0.    ],
                     [ 0.   ,  0.   ,  0.,  0., 0.    ],
                     [ 0.   ,  0.   ,  0.,  0., 0.    ]],

                    [[ 1.0,  0.   ,  0.,  0., 0. ],
                     [ 0.   ,  0.   ,  0.,  0., 0.    ],
                     [ 1.0,  0.   ,  0.,  0., 0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_31(): # Wrap-around test 10 - wrap-around with interpolation on the boundary (corner case), multiple sample points to insert
    data_coord = np.array([[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]])
    data_val = np.array([1, 1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 1.0,  0.   ,  1.0],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 1.0,  0.   ,  1.0]],

                    [[ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ]],

                    [[ 1.0,  0.   ,  1.0],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 1.0,  0.   ,  1.0]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_32():
    small = 1e-10
    data_coord = np.array([[small, small, small], [1-small, 1-small, 1-small]])
    data_val = np.array([1, 1])
    mask = np.ones(len(data_val))

    N_bin = np.array([4, 4, 4])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([2, 2, 2])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="truncate")

    weightout[weightout == 0] = 1
    dataout /= weightout

    ans_dataout = np.array([[[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],],

                             [[0., 0., 0., 0.],
                              [0., 1., 1., 0.],
                              [0., 1., 1., 0.],
                              [0., 0., 0., 0.],],

                             [[0., 0., 0., 0.],
                              [0., 1., 1., 0.],
                              [0., 1., 1., 0.],
                              [0., 0., 0., 0.],],

                             [[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.]]])
    ans_dataout_sum = 8.0

    assert np.sum(np.abs(dataout - ans_dataout)) < 1e-9
    assert np.sum(np.abs(np.sum(dataout) - ans_dataout_sum)) < 1e-9



def test_33():
    small = 1e-10
    data_coord = np.array([[small, small, small], [1-small, 1-small, 1-small]])
    data_val = np.array([1, 1])
    mask = np.ones(len(data_val))

    N_bin = np.array([4, 4, 4])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([2, 2, 2])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, boundary_mode="periodic")

    weightout[weightout == 0] = 1
    dataout /= weightout

    ans_dataout = np.array([[[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],],

                             [[0., 0., 0., 0.],
                              [0., 1., 1., 0.],
                              [0., 1., 1., 0.],
                              [0., 0., 0., 0.],],

                             [[0., 0., 0., 0.],
                              [0., 1., 1., 0.],
                              [0., 1., 1., 0.],
                              [0., 0., 0., 0.],],

                             [[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.]]])
    ans_dataout_sum = 8.0

    assert np.sum(np.abs(dataout - ans_dataout)) < 1e-9
    assert np.sum(np.abs(np.sum(dataout) - ans_dataout_sum)) < 1e-9



def test_34():
    """ Test 1 for the trilinear_insertion_factor function which multiplies 
        a factor onto the insertion weights that Rainier needs for the MCEMC project.
    """

    # Set up input
    data_coord = np.array([[0.1, 0, 0]], dtype=np.float64)
    data_val = np.array([1], dtype=np.float64)

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    densities = np.zeros([3,3,3,2])
    weight_factor = 1.0

    # Do the trilinear insertion
    density.trilinear_insertion_factor(densities=densities, 
                                       weight_factor=weight_factor, 
                                       vectors=data_coord, 
                                       insert_vals=data_val, 
                                       corners=None, 
                                       deltas=None, 
                                       x_min=x_min, 
                                       x_max=x_max)

    # Expected answer
    ans = np.array([[[ 0. ,  0. ,  0. ],
                        [ 0. ,  0. ,  0. ],
                        [ 0. ,  0. ,  0. ]],

                       [[ 0. ,  0. ,  0. ],
                        [ 0. ,  1.0,  0. ],
                        [ 0. ,  0. ,  0. ]],

                       [[ 0. ,  0. ,  0. ],
                        [ 0. ,  1.0,  0. ],
                        [ 0. ,  0. ,  0. ]]])

    # Extract the inserted array and weights from the 4D "density" array
    dataout = densities[:,:,:,0]
    weightout = densities[:,:,:,1]
    w = np.where(weightout != 0)
    data_avg = dataout.copy()
    data_avg[w] /= weightout[w]

    assert np.sum(np.abs(data_avg - ans)) < 1e-9



def test_35():
    """ Test 2 for the trilinear_insertion_factor function which multiplies 
        a factor onto the insertion weights that Rainier needs for the MCEMC project.
    """

    data_coord = np.array([[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]], dtype=np.float64)
    data_val = np.array([1, 1], dtype=np.float64)

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    densities = np.zeros([3,3,3,2])
    weight_factor = 1.0

    # Do the trilinear insertion
    density.trilinear_insertion_factor(densities=densities, 
                                       weight_factor=weight_factor, 
                                       vectors=data_coord, 
                                       insert_vals=data_val, 
                                       corners=None, 
                                       deltas=None, 
                                       x_min=x_min, 
                                       x_max=x_max)
    # Expected answer
    ans = np.array([[[ 1.0,  0.   ,  1.0],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 1.0,  0.   ,  1.0]],

                    [[ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ]],

                    [[ 1.0,  0.   ,  1.0],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 1.0,  0.   ,  1.0]]])


    # Extract the inserted array and weights from the 4D "density" array
    dataout = densities[:,:,:,0]
    weightout = densities[:,:,:,1]
    w = np.where(weightout != 0)
    data_avg = dataout.copy()
    data_avg[w] /= weightout[w]

    assert np.sum(np.abs(data_avg - ans)) < 1e-9



#============================================================================
# There is potentially a bug given by this test code below that needs to be investigated.

'''
def test_36():
    """ Test 3 for the trilinear_insertion_factor function which multiplies 
        a factor onto the insertion weights that Rainier needs for the MCEMC project.
    """

    # data_coord = np.array([[1.0, 1.0, 1.0], \
    #                        [0,0,0]])
    # data_val = np.array([ 1.0,  1.0])
    data_coord = np.array([[1.0, 1.0, 1.0]])
    data_val = np.array([ 2.0])

    N_bin = np.array([2, 2, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([2, 2, 2])

    densities = np.zeros(np.append(N_bin, 2)) # The 2 is because the densities array is now 4D with the fourth dimension storing the weights
    weight_factor = 1.0

    # Do the trilinear insertion
    density.trilinear_insertion_factor(densities=densities.T, 
                                       weight_factor=weight_factor, 
                                       vectors=data_coord.T, 
                                       insert_vals=data_val, 
                                       corners=None, 
                                       deltas=None, 
                                       x_min=x_min, 
                                       x_max=x_max)
    
    yay3
    # Expected answer
    dataout, weightout = utils.trilinear_insert(data_coord, 
                                                data_val, 
                                                x_min, 
                                                x_max, 
                                                N_bin, 
                                                mask=np.ones(len(data_val)), 
                                                boundary_mode="periodic")
    weightout[weightout == 0] = 1
    ans = dataout / weightout

    # Extract the inserted array and weights from the 4D "density" array
    dataout = densities[:,:,:,0]
    weightout = densities[:,:,:,1]
    w = np.where(weightout != 0)
    data_avg = dataout.copy()
    data_avg[w] /= weightout[w]

    print(np.sum(np.abs(data_avg - ans)))
    print(ans)
    print(data_avg)

    assert np.sum(np.abs(data_avg - ans)) < 1e-7


test_36()
'''
#============================================================================




if __name__ == '__main__':
    test_1()

