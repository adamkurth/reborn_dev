import numpy as np

class RadialProfile(object):

	def __init__(self):
		
		self.nBins = None
		self.bins = None
		self.binSize = None
		self.binIndices = None
		self.mask = None
		self.counts = None
		self.qRange = None

	def make_plan(self,PanelList,mask=None,nBins=100,qRange=None):

		''' Use PanelList as template to cache information for radial binning. '''

		self.Qmag = PanelList.Qmag.copy()
		if qRange is None:
			self.minQ = np.min(self.Qmag)
			self.maxQ = np.max(self.Qmag)
			self.qRange = np.array([self.minQ,self.maxQ])
		else:
			self.qRange = qRange.copy()
			self.minQ = qRange[0]
			self.maxQ = qRange[1]
	
		self.nBins = nBins
		self.binSize = self.maxQ / (self.nBins - 1)
		self.bins = (np.arange(0,self.nBins) + 0.5)*self.binSize
		self.binIndices = np.int64(np.floor(self.Qmag / self.binSize))
		if mask is None:
			self.mask = np.ones([len(self.binIndices)])
		else:
			self.mask = mask.copy()
		self.counts = np.bincount(self.binIndices, self.mask, self.nBins)
		self.countsNonZero = self.counts > 0

	def get_profile(self, PanelList, average=True):

		fail = False

		profile = np.bincount(self.binIndices, PanelList.data*self.mask, self.nBins)
		if average:
			profile.flat[self.countsNonZero] /= self.counts.flat[self.countsNonZero]
	
		return profile, fail
