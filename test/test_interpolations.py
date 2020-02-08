from bornagain import utils
import numpy as np

# The usage of N_data and N_pattern in some of the tests below can be interpreted as:
# N_data    : Number of pixels in a detector
# N_pattern : Number of patterns measured from that detector
# And the action of the trilinear_insert() is to insert the values measured by the detector
# into a 3D regularly grided array.

def test_1():
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
                    [ 0. ,  0.9,  0. ],
                    [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                    [ 0. ,  0.1,  0. ],
                    [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


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
                    [ 0.   ,  0.125,  0.125],
                    [ 0.   ,  0.125,  0.125]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.125,  0.125],
                    [ 0.   ,  0.125,  0.125]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    ans = np.array([[[ 0.125,  0.125,  0.   ],
            [ 0.125,  0.125,  0.   ],
            [ 0.   ,  0.   ,  0.   ]],

           [[ 0.125,  0.125,  0.   ],
            [ 0.125,  0.125,  0.   ],
            [ 0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    ans = np.array([[[ 0.12507502,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]]])


    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [ 0.        ,  0.        ,  0.12507502]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [  0.     ,   0.     ,  62.22944],
            [  0.     ,   0.     ,  15.55736]],

           [[  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ]],

           [[  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ],
            [  0.     ,   0.     ,   0.     ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [ 0.        ,  0.076496791,  0.112277353],
            [ 0.        ,  0.105961945,  0.058409726]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.174489971,  0.112437026],
            [ 0.        ,  0.256917323,  0.103009865]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [ 0.  ,  0.25,  0.25,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.25,  0.25,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  1.  ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [ 0.   ,  0.   ,  6.   ,  2.   ],
            [ 0.   ,  0.   ,  0.15 ,  0.25 ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  1.5  ,  0.5  ],
            [ 0.   ,  0.   ,  0.225,  0.375],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [ 0.   ,  0.   ,  6.   ,  2.   ],
            [ 0.   ,  0.   ,  0.15 ,  0.25 ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  1.5  ,  0.5  ],
            [ 0.   ,  0.   ,  0.225,  0.375],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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

    ans = np.array([[[ 0.44481935,  0.07150083,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.03423333,  0.00550271,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]],

       [[ 0.60057795,  0.09653766,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.04622053,  0.00742955,  0.        ,  0.07410725,  2.30167695,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.03272169,  1.01629419,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ]],

       [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.05669123,  1.76075739,
          0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.02503173,  0.7774538 ,
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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
            [ 0.  ,  0.25,  0.25,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.25,  0.25,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]],

           [[ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
    
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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=False)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
                    [ 0.   ,  0.125j,  0.125j],
                    [ 0.   ,  0.125j,  0.125j]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.125j,  0.125j],
                    [ 0.   ,  0.125j,  0.125j]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

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
                    [ 0.   ,  0.45+0.0625j,  0.125j],
                    [ 0.   ,  0.125j,  0.125j]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.05+0.0625j,  0.125j],
                    [ 0.   ,  0.125j,  0.125j]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9



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
                     [ 0. ,  0.9,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.1,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

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

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

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
                     [ 0. ,  0.9 ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.1,  0. ],
                     [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

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
                     [ 0. ,  0.1 ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.,  0. ],
                     [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.9,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

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
                     [ 0. ,  0.1 ,  0. ],
                     [ 0. ,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.,  0. ],
                     [ 0. ,  0. ,  0. ]],

                   [[ 0. ,  0. ,  0. ],
                     [ 0. ,  0.9,  0. ],
                     [ 0. ,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

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
                     [ 0. ,  0. ,  0.1,  0. ,  0. ],
                     [ 0. ,  0. ,  0.,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0.,  0. ,  0. ],
                     [ 0. ,  0.,  0.,  0. ,  0. ],
                     [ 0. ,  0. ,  0.,  0. ,  0. ]],

                    [[ 0. ,  0. ,  0.,  0. ,  0. ],
                     [ 0. ,  0.,  0.9,  0. ,  0. ],
                     [ 0. ,  0. ,  0.,  0. ,  0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_28():
    data_coord = np.array([[1.5, 1.5, 1.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 3])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.125,  0.   ,  0.125],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 0.125,  0.   ,  0.125]],

                    [[ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 0.   ,  0.   ,  0.   ]],

                    [[ 0.125,  0.   ,  0.125],
                     [ 0.   ,  0.   ,  0.   ],
                     [ 0.125,  0.   ,  0.125]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9



def test_29():
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
                    [ 0.   ,  0.125,  0.125],
                    [ 0.   ,  0.125,  0.125]],

                   [[ 0.   ,  0.   ,  0.   ],
                    [ 0.   ,  0.125,  0.125],
                    [ 0.   ,  0.125,  0.125]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

    weightout[weightout == 0] = 1
    dataout /= weightout

    assert np.sum(np.abs(dataout - ans)) < 1e-9



def test_30():
    data_coord = np.array([[1.5, 1.5, 1.5]])
    data_val = np.array([1])
    mask = np.ones(len(data_val))

    N_bin = np.array([3, 3, 5])
    x_min = np.array([-1, -1, -1])
    x_max = np.array([1, 1, 1])

    ans = np.array([[[ 0.25,  0.   ,  0.,  0., 0. ],
                     [ 0.   ,  0.   ,  0.,  0., 0.],
                     [ 0.25,  0.   ,  0.,  0., 0. ]],

                    [[ 0.   ,  0.   ,  0.,  0., 0.    ],
                     [ 0.   ,  0.   ,  0.,  0., 0.    ],
                     [ 0.   ,  0.   ,  0.,  0., 0.    ]],

                    [[ 0.25,  0.   ,  0.,  0., 0. ],
                     [ 0.   ,  0.   ,  0.,  0., 0.    ],
                     [ 0.25,  0.   ,  0.,  0., 0. ]]])

    dataout, weightout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask, wrap_around=True)

    weightout[weightout == 0] = 1
    dataout /= weightout

    print(dataout)

    assert np.sum(np.abs(dataout - ans)) < 1e-9


test_30()