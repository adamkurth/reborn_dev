from functools import wraps
import pkg_resources
import numpy as np
from bornagain import units

atomic_symbols_file = pkg_resources.resource_filename('bornagain.simulate','data/atomic_symbols.csv')
atomic_symbols = np.loadtxt(atomic_symbols_file,usecols=(1,),dtype=np.str_,delimiter=',')
atomic_symbols = np.array([x.strip() for x in atomic_symbols])
henke_data_path = pkg_resources.resource_filename('bornagain.simulate','data/henke')


def memoize(function):
    memo = {}
    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

def atomic_symbols_to_numbers(symbols):

    symbols = np.array(symbols)
    symbols = symbols
    Z = np.zeros([len(symbols)])
    U = np.unique(symbols)
    for u in U:
        w = np.nonzero(atomic_symbols == u.capitalize())
        Z[symbols == u] = w[0]+1
    Z = Z.astype(np.int)
    if len(Z) == 1: Z=Z[0]
    return Z

def atomic_numbers_to_symbols(numbers):
    
    numbers = np.array(numbers,dtype=np.int)
    symbols = atomic_symbols[numbers-1]
    if len(symbols) == 1: symbols = symbols[0] 
    return symbols
    
@memoize
def get_henke_data(atomic_number):
 
    sym = atomic_numbers_to_symbols(atomic_number)
    sym = sym[0]
    table = np.loadtxt(henke_data_path + '/' + sym.lower()+'.nff',skiprows=1)
    data = {}
    data['Atomic Number'] = atomic_number
    data['Element Symbol'] = sym
    data['Photon Energies'] = table[:,0]/units.eV
    data['Scatter Factors'] = table[:,1] + 1j*table[:,2]
    
    return data