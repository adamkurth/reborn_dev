from bornagain import utils
import numpy as np


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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_3():
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_4():
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
    assert np.sum(np.abs(dataout - ans)) < 1e-9


def test_7():
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


    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
    assert np.sum(np.abs(dataout - ans)) < 1e-6


def test_8():
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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
            [ 0.        ,  0.76496791,  1.12277353],
            [ 0.        ,  1.05961945,  0.58409726]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  1.74489971,  1.12437026],
            [ 0.        ,  2.56917323,  1.03009865]]])

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)

    # print(dataout)
    # print(np.sum(dataout))
    # print(np.sum(np.abs(dataout - ans)))


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

    dataout = utils.trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask)
    assert np.sum(np.abs(dataout - ans)) < 1e-9



