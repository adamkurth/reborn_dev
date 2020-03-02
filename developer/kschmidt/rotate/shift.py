import numpy as np
import matplotlib.pyplot as plt

N = 129
L = 10.0
scale = N/L
a = 2.7
sa = a*scale
x = np.linspace(-L/2,L/2,N)
y = np.zeros((N,2),dtype=np.float64)
y[:,0] = np.exp(-x**2)
y[:,1] = np.exp(-(x+3.0)**2)
plt.plot(x,y[:,0])
plt.plot(x,y[:,1])
plt.show()

kfac = np.exp(-1j*2*np.pi/N*sa*np.arange(N))
xfac = np.exp(-1j*np.pi*(1-(N%2)/N)*(np.arange(N)-sa))
yk = np.fft.fft(y,axis=0)
yk = np.fft.fftshift(yk,axes=0)
yk *= kfac[:,np.newaxis]
ys = np.fft.ifft(yk,axis=0)
ys *= xfac[:,np.newaxis]

plt.plot(x,np.real(ys[:,0]))
plt.plot(x,np.imag(ys[:,0]))
plt.plot(x,np.real(ys[:,1]))
plt.plot(x,np.imag(ys[:,1]))
plt.show()
