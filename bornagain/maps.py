"""
Classes related to density or intensity maps.  "Map" means a 3D grid of samples.
"""

import numpy               as np
from scipy.interpolate import RegularGridInterpolator
from bornagain         import utils
#import pyqtgraph           as pg
from scipy.stats       import binned_statistic_dd

class meshTool(object):

	''' Teh meshTool class is supposed to help us work with 3D volumetric density maps.
	    In particular it should help avoid re-thinking the way in which we index voxels
	    over and over again.  Once an instance of this class is created, we can easily
	    generate an array of position vectors corresponding to each voxel, and we can
	    easily find the array index for a given position vector, and we can easily
	    interpolate values, and so on.'''

	def __init__(self,a,n):
		
		''' Initialization input:
		      a is meant to be the physical distance from the center of the cube, to the
		        center of a given face of the cube.
		      n is number of voxels from center of the cube to the face of the cube, not
		        counting the [0,0,0] voxel.
		
		    For clarity, the resulting map will be a cube of size m along an edge, where
		    m = 2*n + 1.  The total number of voxels is m^3'''
	
		self.a = a  # Distance at edge of cube (which is symmetric, size 2*n+1)
		self.n = n  # Number of voxels from center to edge
		self.m = n*2+1 # Number of voxels from edge to edge: 2*n+1
		self.x = np.arange(-n,n+1)*(a/float(n)) # Array of edge values (centers)
		self.s = float((self.m-1)/float(a*2)) # Scale factor for indexing
		self.M = self.m**3 # Total number of voxels
		
		self.binEdges = np.arange(-n-0.5,n+1.5)*(a/float(n)) # Actual bin edges
		
		self._pos = None
		self._posMag = None
		
		
	def zeros(self):
		''' Generate zero array of appropriate size and shape.'''
		return np.zeros([self.m,self.m,self.m])
		
	def reshape(self,dat):
		''' Convert flattened array to 3D array. '''
		dat = dat.reshape((self.m,self.m,self.m))
		return dat
		
	def pos(self):
		''' Return 3xN array of positions corresponding to each voxel in linear format.'''
		if self._pos is None:
			[x,y,z] = np.meshgrid(self.x,self.x,self.x,indexing='ij')
			self._pos = np.array([x.ravel(),y.ravel(),z.ravel()])
			self._pos.flags.writeable = False
		return self._pos
		
	def posMag(self):
		''' Return 1xN array of position vector magnitudes corresponding to each voxel.'''
		if self._posMag is None:
			self._posMag = np.sqrt(np.sum(self.pos()**2,axis=0))
			self._posMag.flags.writeable = False
		return self._posMag
		
	def ind(self):
		''' Return 3xN array of indices corresponding to each voxel.'''
		x = np.arange(0,self.m)
		[x,y,z] = np.meshgrid(x,x,x,indexing='ij')
		return np.array([x.ravel(),y.ravel(),z.ravel()])
		
	def lind(self):
		return np.arange(0,self.m**3)
		
	def pos2ind(self,pos,_round=False,wrap=False):
		''' Convert 3xN array of positions to 3xN array of indices.'''
		ind = pos*self.s + self.n
		if _round:
			ind = np.round(ind).astype(np.int)
		if wrap:
			ind = ind % self.m
		return ind
		
	def ind2pos(self,ind):
		''' Convert 3xN array of indices to 3xN array of positions.s'''
		return (ind - self.n)/self.s
		
	def ind2lind(self,ind):
		''' Convert 3xN array of indices to Nx1 array of linear indices.'''
		i = utils.vecCheck(ind)
		m = utils.vecCheck([self.m**2,self.m,1])
		return m.T.dot(i).astype(np.int)
		
	def lind2ind(self,lind):
		''' Convert Nx1 array of indices to 3xN array of indices.'''
		m = utils.vecCheck([self.m**2,self.m,1])
		return  m.dot(lind).astype(np.int)
		
	def pos2lind(self,pos):
		''' Convert Nx1 array of linear indices to 3xN array of positions.'''
		return self.ind2lind(self.pos2ind(pos)).astype(np.int)
		
	def lind2pos(self,lind):
		''' Convert Nx1 array of linear indices to 3xN array of positions.'''
		return  self.ind2pos(self.lind2ind(lind))
		
	def collect(self,pos,val,_round=False,wrap=False,average=False):
		''' Sum Nx1 array of values into given voxels specified by 3xN position vector
		    list. '''
		ind = np.round(self.pos2ind(pos,_round=False,wrap=False)).astype(np.int)
		dat = self.zeros()
		if average:
			count = self.zeros()
			for i in np.arange(0,ind.shape[1]):
				# This is dangerous -- silently ignore errors...
				if ind[0][i] < 0 or ind[0][i] >= self.m: continue
				if ind[1][i] < 0 or ind[1][i] >= self.m: continue
				if ind[2][i] < 0 or ind[2][i] >= self.m: continue
				dat  [ind[0][i]][ind[1][i]][ind[2][i]] += val[i]
				count[ind[0][i]][ind[1][i]][ind[2][i]] += 1
			dat.flat[count.flat > 0] /= count.flat[count.flat > 0]
		else:
			for i in np.arange(0,ind.shape[1]):
				# This is dangerous -- silently ignore errors...
				if ind[0][i] < 0 or ind[0][i] >= self.m: continue
				if ind[1][i] < 0 or ind[1][i] >= self.m: continue
				if ind[2][i] < 0 or ind[2][i] >= self.m: continue
				dat[ind[0][i]][ind[1][i]][ind[2][i]] += val[i]
		return dat

	def binned_statistic_dd(self,pos,val,bins=None):
		''' This is a wrapper for scipy.stats.binned_statistic_dd().'''
		if bins is None:
			bins = [self.binEdges]*3
		dat = binned_statistic_dd(pos.T,val.ravel(),bins=bins)
		return dat

	def interpolatePos(self,dat,pos):
		''' Return intensities corresponding to 3xN array of positions.  Do linear
		    interpolation.'''
		intp = RegularGridInterpolator((self.x, self.x, self.x), dat, bounds_error=False, fill_value=0)
		return intp(pos)
