"""
Seems to be a duplicate of the "maps" module.  Neither are tested, and will evolve when the need arises.
"""

import numpy as np

class map3d(object):

	'''
	A container for 3D maps.
	'''

	def __init__(self,n=None,edges=None):
			
		n = np.array(n).astype(np.int)
		edges = np.array(edges)	
		
		if len(n.shape) == 0:
			self.n = np.array([n,n,n]).astype(np.int)
		else:
			self.n = np.array(n).astype(np.int)
		
		if len(edges.shape) == 0:
			self.edges = np.zeros([3,2])
			self.edges[:,0] = -edges
			self.edges[:,1] = edges
		elif len(edges.shape) == 1:
			self.edges = np.zeros([3,2])
			self.edges[:,0] = -edges
			self.edges[:,1] = edges
		else:
			self.edges = np.array(edges)
		
		self.pos = None
		
	def positionVectors(self):
	
		if self.pos is not None:
			return self.pos
	
		n = self.n
		e = self.edges
		x0 = np.arange(0,n[0])*(e[0,1]-e[0,0])/float(n[0]-1) + e[0,0]
		x1 = np.arange(0,n[1])*(e[1,1]-e[1,0])/float(n[1]-1) + e[1,0]
		x2 = np.arange(0,n[2])*(e[2,1]-e[2,0])/float(n[2]-1) + e[2,0]
	
		x0, x1, x2 = np.meshgrid(x0,x1,x2)
		
		
		return x1
		
