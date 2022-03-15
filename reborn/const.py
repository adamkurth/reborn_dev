# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

import scipy.constants as const

boltzmann_constant = const.value('Boltzmann constant')
classical_electron_radius = const.value('classical electron radius')
speed_of_light = const.c
planck_constant = const.h
avogadros_number = const.N_A
electron_volt = const.value('electron volt')

h = planck_constant
k = boltzmann_constant
c = speed_of_light
hc = const.h*const.c
N_A = avogadros_number
eV = electron_volt
r_e = classical_electron_radius