import numpy as np
import matplotlib.pyplot as plt
from reborn import detector, source
from reborn.simulate import solutions
pad = detector.PADGeometry(shape=(1000, 1000), distance=0.1, pixel_size=200e-6)
beam = source.Beam(wavelength=1.5e-10)
q = pad.q_mags(beam=beam)
q = np.linspace(0, 10e10, 10000)




def fitup_clark(c):
    xg = np.linspace(0, 0.8, 100)
    fnames = ['clark2010_-37.5.csv', 'clark2010_25.csv', 'clark2010_75.csv']
    temps = np.array([-37.5, 25, 75])+273.15
    filename = fnames[c]
    dat = np.genfromtxt(filename, delimiter=',')
    x = dat[:, 0]
    y = dat[:, 1]*100
    # plt.plot(x, y, '.', label=temps[c])
    fit = np.poly1d(np.polyfit(x, y, 5))
    yf = fit(xg)
    plt.plot(xg, yf, '-', label=temps[c])
    return xg, yf

for i in range(3):
    fitup_clark(i)
# yf = fitup_clark('clark2010_75.csv', xg=xg)
# yf = fitup_clark('clark2010_-37.5.csv', xg=xg)

def fitup_hura(c):
    x, y, t = solutions._get_hura_water_data()
    x /= 1e10
    print(x[0])
    y = y[:, c]
    # plt.plot(x, y, '.', label=t[c])
    _, y, t = solutions._get_hura_water_data_smoothed()
    # x /= 1e10
    y = y[:, c]
    plt.plot(x, y, '-', label=t[c])

# fitup_hura(6)
for i in range(7):
    fitup_hura(i)

plt.legend()
plt.show()