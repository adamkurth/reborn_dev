import numpy as np
from pylab import*
import h5py
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.interpolate import interp1d

def get_radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def get2Dimage_from_radial(radius,radial,n):
    x = np.arange(n)    
    f = interp1d(x, radial)
    return f(radius.flat).reshape(radius.shape)


n=1000; scale=1.
data = np.ones([n,n])
xaxis = np.arange(-n/2,n/2+1,1)*scale
yaxis = np.arange(-n/2,n/2+1,1)*scale
xx, yy = np.meshgrid(xaxis,yaxis)
radius = np.sqrt(xx**2. + yy**2.)
sin = sin(radius/20)
#radius = np.copy(sin)
#print(radius.max)

randomnoise = np.random.rand(n+1,n+1)*10

runFile = "pattern-000001.h5"
f = h5py.File(runFile, 'r')
data = f["data"]
tim = data["ideal"]
tim = np.array(tim)
im = tim.copy()
f.close()

#plt.plot(im)
#plt.show()

fig = plt.figure()

#pattern = sin*50 - radius + randomnoise
ax1 = fig.add_subplot(221)
ax1.set_title('sim ideal')
ax1.imshow(im)

# Calculate radial from 2D original ideal
radial = get_radial_profile(im,[n/2.,n/2.])

ax2 = fig.add_subplot(222)
ax2.set_title('radial from original')
ax2.plot(radial)

# Generate 2D pattern from radial

#padded_radial = np.pad(radial, (0,n-radial.size),  'minimum')
#padded_radial = np.pad(radial, (0,n-radial.size),  'maxmimum)
#padded_radial = np.pad(radial, (0,n-radial.size),  'edge')

#diff=get2Dimage_from_radial(radius,padded_radial,n)

#ax3 = fig.add_subplot(223)
#ax3.set_title('2D from radial')
#ax3.imshow(diff.T,origin='lower',interpolation='nearest')

# Back to radial from fake 2D pattern
#radial2 = get_radial_profile(diff,[n/2.,n/2.])
#ax4 = fig.add_subplot(224)
#ax4.set_title('radial from fake 2d pattern')
#ax4.plot(radial2)

plt.show()
