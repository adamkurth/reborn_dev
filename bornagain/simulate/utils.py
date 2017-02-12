import xraylib             as xr
import numpy               as np
from   bornagain       import units

def atomicScatteringFactors(elements,energy):

	''' Convert elements into complex atomic scattering factors.  Input can be strings or atomic Z numbers.  Energy is, of course, in SI units.'''

	keV = energy*units.keV # Convert from SI to keV
	if len(elements) == 1:
		sample = elements
	else:
		sample = elements[0]
	if type(sample) is str:
		Z = np.array([xr.SymbolToAtomicNumber(e.strip().capitalize()) for e in elements])
	else:
		Z = elements
	fp = np.array([xr.Fi(int(z),keV) for z in Z])
	fpp = np.array([xr.Fii(int(z),keV) for z in Z])
	
	return np.array(Z + fp + 1j*fpp)