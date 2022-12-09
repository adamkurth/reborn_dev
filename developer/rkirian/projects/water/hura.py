import numpy as np
import matplotlib.pyplot as plt
from reborn import detector, source
from reborn.simulate import solutions
pad = detector.PADGeometry(shape=(1000, 1000), distance=0.1, pixel_size=200e-6)
beam = source.Beam(wavelength=1.5e-10)
q = pad.q_mags(beam=beam)
q = np.linspace(0, 10e10, 10000)

# # COMPARE HURA TO CLARK
# def fitup_clark(c):
#     xg = np.linspace(0, 0.8, 100)
#     fnames = ['clark2010_-37.5.csv', 'clark2010_25.csv', 'clark2010_75.csv']
#     temps = np.array([-37.5, 25, 75])+273.15
#     filename = fnames[c]
#     dat = np.genfromtxt(filename, delimiter=',')
#     x = dat[:, 0]
#     y = dat[:, 1]*100
#     plt.plot(x, y, '.', label=temps[c])
#     fit = np.poly1d(np.polyfit(x, y, 5))
#     yf = fit(xg)
#     plt.plot(xg, yf, '-', label=f"{temps[c]:.1f}")
#     return xg, yf
# for i in range(3):
#     fitup_clark(i)
# # yf = fitup_clark('clark2010_75.csv', xg=xg)
# # yf = fitup_clark('clark2010_-37.5.csv', xg=xg)
# def fitup_hura(c):
#     x, y, t = solutions._get_hura_water_data()
#     x /= 1e10
#     print(x[0])
#     y = y[:, c]
#     plt.plot(x, y, '.', label=t[c])
#     _, y, t = solutions._get_hura_water_data_smoothed()
#     # x /= 1e10
#     y = y[:, c]
#     plt.plot(x, y, '-', label=f"{t[c]:.1f}")
# fitup_hura(6)
# for i in range(7):
#     fitup_hura(i)
# plt.legend()
# plt.show()
#
# # WRITE OUT CLARK DATA WITH POLYNOMIAL FITS
# def write_clark_data():
#     xg = np.linspace(0, 0.8, 100)
#     fnames = ['clark2010_-37.5.csv', 'clark2010_25.csv', 'clark2010_75.csv']
#     temps = np.array([-37.5, 25, 75])+273.15
#     dats = []
#     for i in range(3):
#         dat = np.genfromtxt(fnames[i], delimiter=',')
#         x = dat[:, 0]
#         y = dat[:, 1]*100
#         fit = np.poly1d(np.polyfit(x, y, 5))
#         yf = fit(xg)
#         dats.append(yf)
#     dats = np.array(dats)
#     f = open('clark.txt', 'w')
#     f.write(f'Q {temps[0]:.2f} {temps[1]:.2f} {temps[2]:.2f}\n')
#     for i in range(100):
#         f.write(f'{xg[i]:.4f} {dats[0, i]:.4f} {dats[1, i]:.4f} {dats[2, i]:.4f}\n')
#     f.close()
#
# write_clark_data()

qh = np.linspace(1e10, 1.5e10, 50)
qc = np.linspace(0e10, 0.5e10, 50)
qf = np.linspace(0e10, 2e10, 200)
df = qf[1] - qf[0]
qfr = np.arange(1000)*df + df + qf[-1]
wf = np.ones(200)
wf[:50] = np.linspace(0, 1, 50)
wf[100:150] = np.linspace(1, 0, 50)
wf[150:] = 0
wh = np.ones(200) - wf
wh[0:100] = 0
wc = np.ones(200) - wf
wc[50:] = 0
# plt.plot(wh)
# plt.plot(wf)
# plt.plot(wc)
# plt.show()
temps = np.array([1, 11, 25, 44, 55, 66, 77])
Idat = np.zeros((7, 1200))
for i in range(len(temps)):
    Ic = solutions._get_clark_interpolated(qc, temperature=temps[i]+273.16)
    Ih = solutions._get_hura_interpolated(qh, temperature=temps[i]+273.16, smoothed=True)
    fit = np.poly1d(np.polyfit(np.concatenate([qc, qh]), np.concatenate([Ic, Ih]), 5))
    If = fit(qf)
    # plt.plot(qf, If)
    # plt.plot(qc, Ic)
    # plt.plot(qh, Ih)
    Ic = solutions._get_clark_interpolated(qf, temperature=temps[i]+273.16)
    Ih = solutions._get_hura_interpolated(qf, temperature=temps[i]+273.16, smoothed=True)
    If = fit(qf)
    I = Ic*wc + Ih*wh + If*wf
    # plt.plot(qf, I)
    # plt.show()
    Ihr = solutions._get_hura_interpolated(qfr, temperature=temps[i]+273.16, smoothed=True)
    II = np.concatenate([I, Ihr])
    qq = np.concatenate([qf, qfr])
    plt.plot(qq, II)
    III = solutions._get_water_interpolated(qq, temperature=temps[i]+273.16)
    plt.plot(qq, III)
    plt.show()
    Idat[i, :] = II
# f = open('water.dat', 'w')
# f.write("Q 1 11 25 44 55 66 77\n")
# for i in range(1200):
#     f.write(f"{qq[i]/1e10:.4f} {Idat[0,i]:.4f} {Idat[1,i]:.4f} {Idat[2,i]:.4f} {Idat[3,i]:.4f} {Idat[4,i]:.4f} "
#             f"{Idat[5,i]:.4f} {Idat[6,i]:.4f}\n")
# f.close()