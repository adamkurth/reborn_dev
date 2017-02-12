import numpy as np

def cspadReshapePsanaToCheetah(im):

	""" Transform psana cspad numpy array (32,185,388) array to Cheetah 2D array (1480, 1552).  This only works on standard CSPAD.  I'm not aware of a way to do this generically for any CSPAD such as the 2x2 without hard-coding.  Wish I knew how. """

	imlist = []
	for i in range(0,4):
		slist = []
		for j in range(0,8):
			slist.append(im[j+i*8,:,:])
		imlist.append(np.concatenate(slist))
	return np.concatenate(imlist,axis=1)


def cspadToPanelList(dataIn,panelList,geomDict):

	""" Dump the psana cspad numpy array (32,185,388) into a panelList object.  This function assumes that a CrystFEL geometry file has been transformed into a CrystFEL geomDict dictionary. """

	im = cspadReshapePsanaToCheetah(dataIn)
	for p in geomDict['panels']:
		panelList[p['name']].data = im[p['min_ss']:(p['max_ss']+1),p['min_fs']:(p['max_fs']+1)]

	return panelList

