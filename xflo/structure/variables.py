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

class Variables:
    """Hold primitive and derived variables, as well as residuals vectors

    Attributes:
    _rho : np.array(float)
        Density
    _u : np.array(float)
        Velocity in x direction
    _v : np.array(float)
        Velocity in y direction
    _p : np.array(float)
        Pressure
    c : np.array(float)
        Speed of sound
    s : np.array(float)
        Entropy
    mach : np.array(float)
        Mach number
    cp : np.array(float)
        Pressure coefficient
    res_rho : np.array(float)
        Density residuals
    res_u : np.array(float)
        Momentum in x direction residuals
    res_v : np.array(float)
        Momentum in y direction residuals
    res_e : np.array(float)
        Energy residuals
    """
    def __init__(self, nrows):
        # Primitive variables
        self._rho = np.zeros(nrows, dtype=float)
        self._u = np.zeros(nrows, dtype=float)
        self._v = np.zeros(nrows, dtype=float)
        self._p = np.zeros(nrows, dtype=float)

        # Derived variables
        self._c = np.zeros(nrows, dtype=float)
        self._s = np.zeros(nrows, dtype=float)
        self._mach = np.zeros(nrows, dtype=float)
        self._cp = np.zeros(nrows, dtype=float)

        # Residuals
        self._res_rho = np.zeros(nrows, dtype=float)
        self._res_u = np.zeros(nrows, dtype=float)
        self._res_v = np.zeros(nrows, dtype=float)
        self._res_e = np.zeros(nrows, dtype=float)

        # Map names and values
        self._map = {
            'Density': self._rho,
            'VelocityX': self._u,
            'VelocityY': self._v,
            'Pressure': self._p,
            'SpeedOfSound': self._c,
            'Entropy': self._s,
            'Mach': self._mach,
            'PressureCoefficient': self._cp,
            'ResidualsDensity': self._res_rho,
            'ResidualsMomentumX': self._res_u,
            'ResidualsMomentumY': self._res_v,
            'ResidualsEnergy': self._res_e,
        }

    def get(self, name=None):
        """Get variables by name (see class attributes for naming convention)

        Parameters:
        name : str (default: None)
            Name of the variable

        Returns:
        variables : dict(str: np.array(float)) or np.array(float)
            Dictionary of named variables (if name is None), or requested variable array
        """
        if name:
            return self._map[name]
        else:
            return self._map

    def set(self, i, rho, u, v, p, c, s, mach, cp, res_rho, res_u, res_v, res_e):
        """Set variables in sequence (see class attributes for naming convention)

        Parameters:
        i : int
            Row ID
        """
        self._rho[i] = rho
        self._u[i] = u
        self._v[i] = v
        self._p[i] = p
        self._c[i] = c
        self._s[i] = s
        self._mach[i] = mach
        self._cp[i] = cp
        self._res_rho[i] = res_rho
        self._res_u[i] = res_u
        self._res_v[i] = res_v
        self._res_e[i] = res_e
