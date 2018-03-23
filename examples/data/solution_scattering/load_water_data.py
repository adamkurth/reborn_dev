import numpy as np
import matplotlib.pyplot as plt

file_name = 'water_scattering_data.txt'

with open(file_name, 'r') as f:
    h = f.readline()
h = h.split()
temperatures = h[1:-1]
temperatures = np.array([float(v) for v in temperatures])

d = np.loadtxt(file_name, skiprows=1)
Q = d[:,0]
errors = d[:,-1]
intensities = d[:,1:-1]
for i in range(1,len(temperatures)):
    plt.plot(Q,np.log10(intensities[:,i]), label=('%2g $^\circ$C' % temperatures[i-1]))

plt.legend()
plt.xlabel('Q')
plt.ylabel('Intensity')
plt.show()
print(temperatures)
print(Q)