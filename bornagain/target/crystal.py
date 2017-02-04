''' Basic utilities for reading in atom coordinates from PDB files, viewing coordinates, 
    gathering symmetry operations for space groups, etc. '''


import spglib                as sg
import xraylib               as xr
import numpy                 as np
import matplotlib.pyplot     as plt
# import pyqtgraph             as pg
# import pyqtgraph.opengl      as gl
from   bornagain         import units, utils
from   numpy             import sin,cos,sqrt


class structure(object):

	''' Stuff needed to deal with an atomistic crystal structure. 
	
	    A note on coordinate transformations and the orthogonalization matrix O:
	      Convert from fractional coordinates x to real-space coordinates r:
	        r = cryst.O.dot(x)
	      Convert from real-space coordinates r to fractional coordinates x:
	        x = cryst.Oinv.dot(r)
	      Convert from q-space to h-space:
	        h = cryst.O.T.dot(q)
	'''
	
	def __init__(self,pdbFilePath=None):
	
		self.r = None     # Atomic coordinates (3xN array)
		self._x = None    # Fractional coordinates (3xN array)
		self.O = None     # Orthogonalization matrix (3x3 array)
		self.Oinv = None  # Inverse orthogonalization matrix (3x3 array)
		self.A = None
		self.Ainv = None
		self.T = None     # Translation vector that goes with orthogonalization matrix (??)
		self.elements = None # Atomic element symbols
		self.Z = None     # Atomic numbers
		self.spaceGroupNumber = None # Space group number in the International Tables
		self.hermannMauguinSymbol = None # Spacegroup Hermann Mauguin symbol
		self.a = None     # Lattice constant
		self.b = None     # Lattice constant
		self.c = None     # Lattice constant
		self.alpha = None # Lattice angle
		self.beta = None  # Lattice angle
		self.gamma = None # Lattice angle
		self.V = None     # Unit cell volume
		self.nAtoms = None # Number of atoms
		self.nMolecules = None # Number of molecules per unit cell
		self.symOps = None # Symmetry operations for fractional coords
		
		if pdbFilePath is not None:
			self.loadPDB(pdbFilePath)
		
		
	def loadPDB(self,pdbFilePath):
		''' Populate all the attributes from a PDB file. '''
		parsePDB(pdbFilePath,self)
		
	def setCell(self,a,b,c,alpha,beta,gamma):
		''' Set the unit cell and all quantities derived from the unit cell. '''
		
		al = alpha
		be = beta
		ga = gamma
		
		self.a = a
		self.b = b
		self.c = c
		self.alpha = al
		self.beta = be
		self.gamma = ga
		
		V = a*b*c*sqrt(1-cos(al)**2-cos(be)**2-cos(ga)**2+2*cos(al)*cos(be)*cos(ga))
		O = np.array([[a   , b*cos(ga) , c*cos(be)                            ],
					  [0   , b*sin(ga) , c*(cos(al)-cos(be)*cos(ga))/sin(ga)  ],
					  [0   , 0         , V/(a*b*sin(ga))                      ]])
		Oinv = np.array([[1/a , -cos(ga)/(a*sin(ga)) , 0                    ],
						 [0   , 1/(b*sin(ga))        , 0                    ],
						 [0   , 0                    , a*b*sin(ga)/V        ]])
		self.O = O
		self.Oinv = Oinv
		self.A = Oinv.T.copy()
		self.Ainv = O.T.copy()
		self.V = V
			
	@property
	def x(self):
		''' Fractional coordinates of atoms. '''
		if self._x is None:
			self._x = self.Oinv.dot(self.r)
		return self._x
		
		
		

def parsePDB(pdbFilePath,crystalStruct=None):

	''' Return a structure object with PDB information. '''

	maxAtoms = int(1e5)
	r = np.zeros([3,maxAtoms])
	elements = []
	atomIndex = int(0)
	if crystalStruct is None:
		cryst = structure()
	else:
		cryst = crystalStruct
	SCALE = np.zeros([3,4])
	
	with open(pdbFilePath) as pdbfile:
		
		for line in pdbfile:
		
			# This is the inverse of the "orthogonalization matrix"  along with 
			# translation vector.  See Rupp for an explanation.
			
			if line[:5] == 'SCALE':
				n = int(line[5]) - 1
				SCALE[n,0] = float(line[10:20])
				SCALE[n,1] = float(line[20:30])
				SCALE[n,2] = float(line[30:40])
				SCALE[n,3] = float(line[45:55])
		
			# The crystal lattice and symmetry
		
			if line[:6] == 'CRYST1':
				cryst1 = line
				# As always, everything in our programs are in SI units.  
				# PDB files use angstrom units.
				a = float(cryst1[6:15])*1e-10
				b = float(cryst1[15:24])*1e-10
				c = float(cryst1[24:33])*1e-10
				# And of course degrees are converted to radians (though we loose the perfection 
				# of rational quotients like 360/4=90...)
				al = float(cryst1[33:40])*np.pi/180.0
				be = float(cryst1[40:47])*np.pi/180.0
				ga = float(cryst1[47:54])*np.pi/180.0
			
				cryst.setCell(a,b,c,al,be,ga)
				spcgrp = cryst1[55:66].strip()
			
			if line[:6] == 'ATOM  ' or line[:6] == "HETATM":
				r[0,atomIndex] = float(line[30:38])*1e-10
				r[1,atomIndex] = float(line[38:46])*1e-10
				r[2,atomIndex] = float(line[46:54])*1e-10
				elements.append(line[76:78].strip().capitalize())
				z = xr.SymbolToAtomicNumber(elements[atomIndex])
				atomIndex += 1
				
			if atomIndex == maxAtoms:
				r = np.append(r,np.zeros([3,maxAtoms]),axis=1)
		
	# Truncate atom list since we pre-allocated extra memory
	nAtoms = atomIndex
	r = r[:,:nAtoms]
	elements = elements[:nAtoms]
	
	T = SCALE[:,3]
	
	cryst.r = utils.vecCheck(r)
	cryst.T = utils.vecCheck(T)
	cryst.elements = elements
	cryst.Z = np.array([xr.SymbolToAtomicNumber(e) for e in elements])
	cryst.spaceGroupNumber = hermannMauguinToNumber(spcgrp)
	cryst.hermannMauguinSymbol = spcgrp
	cryst.nAtoms = nAtoms
	cryst.symOps = sg.get_symmetry_from_database(hermannMauguinToNumber(spcgrp,hall=True))
	cryst.nMolecules = len(cryst.symOps['rotations'])
	cryst.symRs = [R for R in cryst.symOps['rotations']]
	cryst.symRinvs = [np.linalg.inv(R) for R in cryst.symOps['rotations']]
	cryst.symTs = [utils.vecCheck(T) for T in cryst.symOps['translations']]
	
	return cryst


#def viewClusterAtoms(cryst,occupancies=None):
#
#	''' Simple 3D viewer to see atomic coordinates and unit cell.  Not very featureful,
#	    but works as a basic sanity check for transformations.'''
#
#	if occupancies is None:
#		occupancies = [[np.zeros([3])] for i in range(0,cryst.nMolecules)]

#	app=pg.QtGui.QApplication([])
#	w = gl.GLViewWidget()
#	scl = 1e10
#	siz = 1

	# Plot each symmetry partner in different colors
#	for i in range(0,cryst.nMolecules):
#		print(i)
#		R = cryst.symRs[i]
#		T = cryst.symTs[i]
#		for Tp in occupancies[i]:
#			Tp = utils.vecCheck(Tp)
#			ff = cryst.O.dot(R.dot(cryst.x)+T+Tp)
#			plt = gl.GLScatterPlotItem(pos=ff.T*scl,color=utils.glColors[i+2], size=siz)
#			w.addItem(plt)

	# Plot points at the origin and lattice vector tips
#	o = np.array([0,0,0])
#	tips = [cryst.O.dot(np.array([1,0,0])),
#			cryst.O.dot(np.array([0,1,0])),
#			cryst.O.dot(np.array([0,0,1]))]

	# Draw the unit cell in white
#	wid=1
#	C=pg.glColor([255,255,255])
#	for i in range(0,3):
#		plt = gl.GLLinePlotItem(pos=np.array([o,tips[i]])*scl,mode='lines',width=wid,color=C)
#		w.addItem(plt)
#		plt = gl.GLLinePlotItem(pos=(np.array([o,tips[i]])+tips[(i+1)%3])*scl,mode='lines',width=wid,color=C)
#		w.addItem(plt)
#		plt = gl.GLLinePlotItem(pos=(np.array([o,tips[i]])+tips[(i+2)%3])*scl,mode='lines',width=wid,color=C)
#		w.addItem(plt)
#		plt = gl.GLLinePlotItem(pos=(np.array([o,tips[i]])+tips[(i+1)%3]+tips[(i+2)%3])*scl,mode='lines',width=wid,color=C)
#		w.addItem(plt)

	# Draw the basis vectors in color
#	for i in range(0,3):
#		plt = gl.GLLinePlotItem(pos=np.array([o,tips[i]])*scl,mode='lines',width=5,color=utils.glColors[i+2])
#		w.addItem(plt)

#	w.setCameraPosition(distance=1000)
#	w.show()
#	pg.QtGui.QApplication.exec_()



def hermannMauguinToNumber(hm,hall=False):

	''' This is needed to convert the Hermann Mauguin space group specifications in PDB files to the space group number or the "Hall number" that is used by the spglib module.'''

	sgnum = [1,2,3,3,3,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,16,17,17,17,18,18,18,19,20,20,20,21,21,21,22,23,24,25,25,25,26,26,26,26,26,26,27,27,27,28,28,28,28,28,28,29,29,29,29,29,29,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,33,33,33,33,33,33,34,34,34,35,35,35,36,36,36,36,36,36,37,37,37,38,38,38,38,38,38,39,39,39,39,39,39,40,40,40,40,40,40,41,41,41,41,41,41,42,42,42,43,43,43,44,44,44,45,45,45,46,46,46,46,46,46,47,48,48,49,49,49,50,50,50,50,50,50,51,51,51,51,51,51,52,52,52,52,52,52,53,53,53,53,53,53,54,54,54,54,54,54,55,55,55,56,56,56,57,57,57,57,57,57,58,58,58,59,59,59,59,59,59,60,60,60,60,60,60,61,61,62,62,62,62,62,62,63,63,63,63,63,63,64,64,64,64,64,64,65,65,65,66,66,66,67,67,67,67,67,67,68,68,68,68,68,68,68,68,68,68,68,68,69,70,70,71,72,72,72,73,73,74,74,74,74,74,74,75,76,77,78,79,80,81,82,83,84,85,85,86,86,87,88,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,125,126,126,127,128,129,129,130,130,131,132,133,133,134,134,135,136,137,137,138,138,139,140,141,141,142,142,143,144,145,146,146,147,148,148,149,150,151,152,153,154,155,155,156,157,158,159,160,160,161,161,162,163,164,165,166,166,167,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,201,202,203,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,222,223,224,224,225,226,227,227,228,228,229,230]
	hmsym = ['P 1','P -1','P 2 ','P 2 ','P 2 ','P 21 ','P 21 ','P 21 ','C 2 ','C 2 ','C 2 ','C 2 ','C 2 ','C 2 ','C 2 ','C 2 ','C 2 ','P m ','P m ','P m ','P c ','P c ','P c ','P c ','P c ','P c ','P c ','P c ','P c ','C m ','C m ','C m ','C m ','C m ','C m ','C m ','C m ','C m ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','C c ','P 2/m ','P 2/m ','P 2/m ','P 21/m ','P 21/m ','P 21/m ','C 2/m ','C 2/m ','C 2/m ','C 2/m ','C 2/m ','C 2/m ','C 2/m ','C 2/m ','C 2/m ','P 2/c ','P 2/c ','P 2/c ','P 2/c ','P 2/c ','P 2/c ','P 2/c ','P 2/c ','P 2/c ','P 21/c ','P 21/c ','P 21/c ','P 21/c ','P 21/c ','P 21/c ','P 21/c ','P 21/c ','P 21/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','C 2/c ','P 2 2 2','P 2 2 21','P 21 2 2','P 2 21 2','P 21 21 2','P 2 21 21','P 21 2 21','P 21 21 21','C 2 2 21','A 21 2 2','B 2 21 2','C 2 2 2','A 2 2 2','B 2 2 2','F 2 2 2','I 2 2 2','I 21 21 21','P m m 2','P 2 m m','P m 2 m','P m c 21','P c m 21','P 21 m a','P 21 a m','P b 21 m','P m 21 b','P c c 2','P 2 a a','P b 2 b','P m a 2','P b m 2','P 2 m b','P 2 c m','P c 2 m','P m 2 a','P c a 21','P b c 21','P 21 a b','P 21 c a','P c 21 b','P b 21 a','P n c 2','P c n 2','P 2 n a','P 2 a n','P b 2 n','P n 2 b','P m n 21','P n m 21','P 21 m n','P 21 n m','P n 21 m','P m 21 n','P b a 2','P 2 c b','P c 2 a','P n a 21','P b n 21','P 21 n b','P 21 c n','P c 21 n','P n 21 a','P n n 2','P 2 n n','P n 2 n','C m m 2','A 2 m m','B m 2 m','C m c 21','C c m 21','A 21 m a','A 21 a m','B b 21 m','B m 21 b','C c c 2','A 2 a a','B b 2 b','A m m 2','B m m 2','B 2 m m','C 2 m m','C m 2 m','A m 2 m','A b m 2','B m a 2','B 2 c m','C 2 m b','C m 2 a','A c 2 m','A m a 2','B b m 2','B 2 m b','C 2 c m','C c 2 m','A m 2 a','A b a 2','B b a 2','B 2 c b','C 2 c b','C c 2 a','A c 2 a','F m m 2','F 2 m m','F m 2 m','F d d 2','F 2 d d','F d 2 d','I m m 2','I 2 m m','I m 2 m','I b a 2','I 2 c b','I c 2 a','I m a 2','I b m 2','I 2 m b','I 2 c m','I c 2 m','I m 2 a','P m m m','P n n n','P n n n','P c c m','P m a a','P b m b','P b a n','P b a n','P n c b','P n c b','P c n a','P c n a','P m m a','P m m b','P b m m','P c m m','P m c m','P m a m','P n n a','P n n b','P b n n','P c n n','P n c n','P n a n','P m n a','P n m b','P b m n','P c n m','P n c m','P m a n','P c c a','P c c b','P b a a','P c a a','P b c b','P b a b','P b a m','P m c b','P c m a','P c c n','P n a a','P b n b','P b c m','P c a m','P m c a','P m a b','P b m a','P c m b','P n n m','P m n n','P n m n','P m m n','P m m n','P n m m','P n m m','P m n m','P m n m','P b c n','P c a n','P n c a','P n a b','P b n a','P c n b','P b c a','P c a b','P n m a','P m n b','P b n m','P c m n','P m c n','P n a m','C m c m','C c m m','A m m a','A m a m','B b m m','B m m b','C m c a','C c m b','A b m a','A c a m','B b c m','B m a b','C m m m','A m m m','B m m m','C c c m','A m a a','B b m b','C m m a','C m m b','A b m m','A c m m','B m c m','B m a m','C c c a','C c c a','C c c b','C c c b','A b a a','A b a a','A c a a','A c a a','B b c b','B b c b','B b a b','B b a b','F m m m','F d d d','F d d d','I m m m','I b a m','I m c b','I c m a','I b c a','I c a b','I m m a','I m m b','I b m m','I c m m','I m c m','I m a m','P 4','P 41','P 42','P 43','I 4','I 41','P -4','I -4','P 4/m','P 42/m','P 4/n','P 4/n','P 42/n','P 42/n','I 4/m','I 41/a','I 41/a','P 4 2 2','P 4 21 2','P 41 2 2','P 41 21 2','P 42 2 2','P 42 21 2','P 43 2 2','P 43 21 2','I 4 2 2','I 41 2 2','P 4 m m','P 4 b m','P 42 c m','P 42 n m','P 4 c c','P 4 n c','P 42 m c','P 42 b c','I 4 m m','I 4 c m','I 41 m d','I 41 c d','P -4 2 m','P -4 2 c','P -4 21 m','P -4 21 c','P -4 m 2','P -4 c 2','P -4 b 2','P -4 n 2','I -4 m 2','I -4 c 2','I -4 2 m','I -4 2 d','P 4/m m m','P 4/m c c','P 4/n b m','P 4/n b m','P 4/n n c','P 4/n n c','P 4/m b m','P 4/m n c','P 4/n m m','P 4/n m m','P 4/n c c','P 4/n c c','P 42/m m c','P 42/m c m','P 42/n b c','P 42/n b c','P 42/n n m','P 42/n n m','P 42/m b c','P 42/m n m','P 42/n m c','P 42/n m c','P 42/n c m','P 42/n c m','I 4/m m m','I 4/m c m','I 41/a m d','I 41/a m d','I 41/a c d','I 41/a c d','P 3','P 31','P 32','R 3','R 3','P -3','R -3','R -3','P 3 1 2','P 3 2 1','P 31 1 2','P 31 2 1','P 32 1 2','P 32 2 1','R 3 2','R 3 2','P 3 m 1','P 3 1 m','P 3 c 1','P 3 1 c','R 3 m','R 3 m','R 3 c','R 3 c','P -3 1 m','P -3 1 c','P -3 m 1','P -3 c 1','R -3 m','R -3 m','R -3 c','R -3 c','P 6','P 61','P 65','P 62','P 64','P 63','P -6','P 6/m','P 63/m','P 6 2 2','P 61 2 2','P 65 2 2','P 62 2 2','P 64 2 2','P 63 2 2','P 6 m m','P 6 c c','P 63 c m','P 63 m c','P -6 m 2','P -6 c 2','P -6 2 m','P -6 2 c','P 6/m m m','P 6/m c c','P 63/m c m','P 63/m m c','P 2 3','F 2 3','I 2 3','P 21 3','I 21 3','P m 3','P n 3','P n 3','F m 3','F d 3','F d 3','I m 3','P a 3','I a 3','P 4 3 2','P 42 3 2','F 4 3 2','F 41 3 2','I 4 3 2','P 43 3 2','P 41 3 2','I 41 3 2','P -4 3 m','F -4 3 m','I -4 3 m','P -4 3 n','F -4 3 c','I -4 3 d','P m -3 m','P n -3 n','P n -3 n','P m -3 n','P n -3 m','P n -3 m','F m -3 m','F m -3 c','F d -3 m','F d -3 m','F d -3 c','F d -3 c','I m -3 m','I a -3 d']
	
	foundMatch = False
	for i,sym in enumerate(hmsym):
		if sym == hm.strip():
			foundMatch = True
			break

	if foundMatch:
		if hall:
			sg = i+1
		else:
			sg = sgnum[i]
		return sg
	else:
		return None
