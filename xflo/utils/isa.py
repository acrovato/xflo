# -*- coding: utf-8 -*-

# xFlo
# Copyright (C) 2025 Adrien Crovato
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def eval(flt_lvl):
    """Compute density and pressure of International Standard Atmosphere

    Parameters:
    flt_lvl : float
        Flight level

    Returns:
    rho, p : tuple(float)
        Density and pressure
    """
    # Constants
    r = 287.058 # perfect gas constant
    rho0 = 1.225 # SL density (kg/m^3)
    t0 = 288.15 # SL temperature (K)
    ht = 11e3 # tropopause height (m)
    htp = 6.3e3 # height scale of the exponential fall at tropopause (m)
    g = 9.81 # acceleration due to gravity (m/s^2)
    l = 0.0065 # temperature lapse rate (K/m)

    # Convert flight level to altitude
    h = flt_lvl * 100 * 0.3048

    # Compute density and temperature
    # Troposhpere
    if h < ht:
        t = t0 - l * h
        rho = rho0 * np.power(1 - l * h / t0, g / (r * l) - 1)
    # Tropopause
    elif h < 20e3:
        t = t0 - l * ht
        rho = rho0 * np.power(1 - l * ht / t0, g / (r * l) - 1) * np.exp(-(h - ht) / htp)
    else:
        raise RuntimeError('MatchedIsaFluid: altitude higher than 65000 ft (20 km)!')

    # Compute pressure
    p = rho * r * t

    return rho, p
