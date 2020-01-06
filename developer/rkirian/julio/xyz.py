#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:14:04 2019

@author: Julio Candanedo
"""

import numpy as np
import math
import scipy.stats as sps
import matplotlib.pyplot as plt
import time

def Bohr(x): ### Given Angstrom get Bohr
    return 1.88973*x
def fs(x): ### Given time in a.u. get fs
    return x/41.341447 #
def Angstrom(x): ### Given Bohr get Angstrom
    return x/1.88973 #

######################## Simulation Parameters
Temperature = 85
Temperature = Temperature/315774.64 ## a.u. temp
T = 1500 ## in units of time step in a.u. tested up to 250 ps!!! Tested in 3d version 2.5 ns!!! 250 ns with 4 Ar! we ran 25 ps for 714 Ar atoms!
dt = 150 ## in ?? a.u? or fs? must be a.u. if used in a.u. formulas!!
time_array = np.linspace(0., T, num=int(T/dt))
Total_CPU = 0
gamma = 0.001

Z_dictonary = np.array(['e ', 'H ', 'He', 
                        'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',
                        'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])

def xyz_reader(name): #currentlogfile = 'Ar_1000.xyz'#'Ne_1000.xyz'#'Na_7.xyz'#'Na_1000.xyz'# #currentlogfile = 'Xe_1Million.xyz' #currentlogfile = 'ArTest20.xyz' #ArTest.xyz

    file = open(name,'r')
    lines = file.readlines()
    lines.pop(0)
    lines.pop(0)
    
    Z = np.array([])
    x = np.array([])
    y = np.array([])
    z = np.array([])

    for element in lines:
        a_line_in_lines = element.split()
        element_number = np.where(Z_dictonary==(element[0] + element[1]))
    
        Z = np.append(Z, element_number[0], axis = 0)
        x = np.append(x, [float(a_line_in_lines[1])], axis = 0)
        y = np.append(y, [float(a_line_in_lines[2])], axis = 0)
        z = np.append(z, [float(a_line_in_lines[3])], axis = 0)
    file.close()
    
    return np.stack((Z, x, y, z))

# def output_xyz(Z, x, y, z):
#     composition_Z = np.unique(Z, return_counts=True)[0]
#     composition_N = np.unique(Z, return_counts=True)[1]
#     word = ""
#     for index, element in enumerate(composition_Z):
#         word = word + " " + Z_dictonary[element] + str(composition_N[index])
#     word = word + ".xyz"
#     word = word.replace(" ", "")
#
#     export_xyz = open(word,"w+")
#     export_xyz.write(str(len(Z)) + "\r\n")
#     export_xyz.write("\r\n")
#     for i in range(len(Z)):
#         export_xyz.write(str(Z_dictonary[Z[i]]) + " " + str(np.around(Angstrom(x[i]),6)) + " " + str(np.around(Angstrom(y[i]),6)) + " " + str(np.around(Angstrom(z[i]),6)) + "\r\n")
#     export_xyz.close()
#
#     return None

######################## Initialize Coordinates
# Zxyz = xyz_reader('Ar_1000.xyz') # ('Ar_200.xyz')
#
# Z = Zxyz[0]
# Z = Z.astype(int) ### convert to an integer numpy array
# Rx_old = Zxyz[1]
# Ry_old = Zxyz[2]
# Rz_old = Zxyz[3]
