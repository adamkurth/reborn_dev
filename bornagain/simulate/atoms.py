from functools import wraps
import pkg_resources
import numpy as np
from bornagain import units
#from astropy.units import photon

atomic_symbols_file = pkg_resources.resource_filename('bornagain.simulate','data/atomic_symbols.csv')
atomic_symbols = np.loadtxt(atomic_symbols_file,usecols=(1,),dtype=np.str_,delimiter=',')
atomic_symbols = np.array([x.strip() for x in atomic_symbols])
henke_data_path = pkg_resources.resource_filename('bornagain.simulate','data/henke')


def memoize(function):
    
    """ 
    This is a function decorator for caching results from a function, to avoid
    excessive computation or reading from disk.  Search the web for more details of how
    this works.  It is a standard practice in Python 2.
    """ 
    
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

    """
    Convert atomic symbols (such as C, N, He, Be) to atomic numbers.
    
    Input:
    symbols: A single scalar or list-like object containing strings with atomic symbols. 
             This is not sensitive to case.
             
    Output:
    numbers: A numpy array of atomic numbers (integer data type) 
    """

    symbols = np.array(symbols)
    if symbols.ndim > 0:
        Z = np.zeros([len(symbols)])
        U = np.unique(symbols)
        for u in U:
            w = np.nonzero(atomic_symbols == u.capitalize())
            Z[symbols == u] = w[0]+1
            Z = Z.astype(np.int)
    else:
        print(symbols)
        w = np.nonzero(atomic_symbols == np.str_(symbols).capitalize())
        Z = w[0] + 1
    
    return Z

def atomic_numbers_to_symbols(numbers):
    
    """
    Convert atomic numbers to atomic symbols (such as C, N, He, Ca)
    
    Input:
    numbers: A list-like object containing atomic numbers (integers)
    
    Output:
    symbols: A numpy string array containing atomic element symbols (such as H, He, Be)
    """ 
    
    numbers = np.array(numbers,dtype=np.int)
    symbols = atomic_symbols[numbers-1]
    if len(symbols) == 1: symbols = symbols[0] 
    return symbols
    
@memoize
def get_henke_data(atomic_number):
 
    """ 
    Load Henke scattering factor data from disk, cache for future retrieval.  The Henke
    data was gathered from the Center for X-Ray Optics web page at the Lawrence Berkeley 
    National Laboratory.  There is a publication with details (reference to be added).  
    
    Input:
    atomic_number: The atomic number of interest (scalar, not an array)
    
    Output:
    data: A dictionary with the following fields:
            'Atomic Number'  : Atomic number of the element
            'Element Symbol' : Standard element symbol (e.g. He)
            'Photon Energy'  : A numpy array of photon energies (SI units)
            'Scatter Factor' : A numpy array of complex atomic scattering factors that 
                               correspond to the photon energies above.
    """
 
    sym = atomic_numbers_to_symbols(atomic_number)
    table = np.loadtxt(henke_data_path + '/' + sym.lower()+'.nff',skiprows=1)
    data = {}
    data['Atomic Number'] = atomic_number
    data['Element Symbol'] = sym
    data['Photon Energy'] = table[:,0]/units.eV
    data['Scatter Factor'] = table[:,1] + 1j*table[:,2]
    
    return data

def get_scattering_factors(atomic_numbers, photon_energy):
    
    """
    Get the atomic scattering factors for a list of atomic numbers and for a given photon
    energy.
    
    Input:
    atomic_numbers: A list-like object containing atomic numbers (integers)
    photon_energy:  A scalar value representing the photon energy (in SI units of course)
    
    Output:
    scattering_factors: A numpy array of complex scattering factors correspondig to the 
                        input atomic numbers and photon_energy.
    """
    
    # TODO: finish off this function, which should use get_henke_data()
    
    Z = np.array(atomic_numbers)
    f = np.zeros([len(Z)],dtype=np.complex)
    U = np.unique(Z)
    for z in U:
        dat = get_henke_data(z)
        E = dat['Photon Energy']
        fdat = dat['Scatter Factor']
        idx = (np.abs(E - photon_energy)).argmin()
        w = np.where(Z == z)
        f[w] = fdat[idx]
    return f
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
