import numpy
import bornagain as ba
import bornagain.simulate.clcore as core

class McCore(object):

	r"""
	A GPU-based simulation utility that generates q_vectors in a Monte Carlo style and returns intensities.
	"""

	# Input parameters and objects

	r = None
	f = None
	detector = None
	beam = None
	I = None


	cl_group_size = 32
	cl_double_precision = False
	clcore = None
	seed = None

	def __init__(self):
		
	
	def load_beam(self, beam)
		r"""
		Loads a Beam object.
		"""
		self.beam = beam

	def load_detector(self, pad)
		r"""
		Loads a PADGeometry object.
		"""
		self.detector = pad

	def cl_init
		self.clcore = core.ClCore(group_size=self.cl_group_size, double_precision=self.cl_double_precision)	

	def mcq(wavelength, wavelength_fwhm, div_angle_fwhm, seed, q):
		r"""
		Generates a set of q vectors by jittering the input q vectors using wavelength and full divergence angle. Considers divergence angle and spectral dispersion.
		"""
		q_new = q
		B = ba.utils.random_beam_vector(div_angle_fwhm)
		if(wavelength == None):
			w = wavelength
		else:
			w = np.random.normal(wavelength, wavelength_fwhm / 2.354
		
