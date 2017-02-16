'''
Created on Feb 14, 2017

@author: rkirian
'''

import pkg_resources

import numpy as np

atomic_symbols_file = pkg_resources.resource_filename('bornagain.simulate','data/atomic_symbols.csv')
#atomic_numbers = np.loadtxt(atomic_symbols_file,usecols=(0,),dtype=np.int,delimiter=',')
atomic_symbols = np.loadtxt(atomic_symbols_file,usecols=(1,),dtype=np.str_,delimiter=',')
atomic_symbols = np.array([x.strip() for x in atomic_symbols])

henke_data_path = pkg_resources.resource_filename('bornagain.simulate','data/henke_data')


def atomic_symbols_to_numbers(symbols):

    symbols = np.array(symbols)
    symbols = symbols
    Z = np.zeros([len(symbols)])
    U = np.unique(symbols)
    for u in U:
        w = np.nonzero(atomic_symbols == u.capitalize())
        Z[symbols == u] = w[0]+1

    return Z

def atomic_numbers_to_symbols(numbers):
    
    numbers = np.array(numbers)
    return atomic_symbols[numbers-1]
    

# def get_henke_data(atomic_number):
# 
#     elem = get_atomic_numbers
#     np.readtxt()
#     return 