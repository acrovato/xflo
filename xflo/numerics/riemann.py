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

from .flux import Flux
import numpy as np

class Riemann(Flux):
    """Utilities for solving Riemann problem
    Riemann Solvers and Numerical Methods for Fluid Dynamics, E.F. Toro,
    Springer, 2009
    https://link.springer.com/book/10.1007/b79761
    """
    def _reconstruct_states(self, s0, s1, ds0, ds1, n, d):
        """Reconstruct states at edge
        Note: since the normal is inward w.r.t. owner cell, 0 is right and 1 is left

        Parameters:
        s0 : np.array(float), size: (4)
            Conservatives variables in the edge's owner cell (right)
        s1 : np.array(float), size: (4)
            Conservatives variables in the edge's neighboor cell (left)
        ds0 : np.array(float), size: (4, 2)
            Gradient of conservatives variables in the edge's owner cell (right)
        ds1 : np.array(float), size: (4, 2)
            Gradient of conservatives variables in the edge's neighboor cell (left)
        n : np.array(float), size: (2)
            Edge unit normal vector
        d : float
            Distance between owner and neighboor cells linked to edge

        Returns:
        s0, s1 : tuple(np.array(float), size: (4))
            Right and left states
        """
        s0 = s0 - ds0.dot(n) * 0.5 * d
        s1 = s1 + ds1.dot(n) * 0.5 * d
        return s0, s1

    def _eval_vars_flux(self, s):
        """Evaluate primitive variables and fluxes in cell

        Parameters:
        s : np.array(float), size: (4)
            Conservatives variables vector

        Returns:
        rho, q, p, c, h, f : tuple(float, ..., np.array(float), size: (4, 2))
            Density, velocity, pressure, speed of sound, enthalpy and fluxes
        """
        # Compute left/right primitives, speed of sound and enthalpy
        rho, q, p = self._flu.eval_primitive(s)
        c = self._flu.eval_speed_sound(rho, p)
        h = self._flu.eval_enthalpy(rho, q, p)
        # Compute left/right fluxes
        f = self._flu.compute_flux(rho, q, p)
        return rho, q, p, c, h, f

    def _eval_roe_average(self, rho0, rho1, q0, q1, h0, h1):
        """Evaluate Roe's averaged variables

        Parameters:
        rho0: float
            Density in owner cell (right)
        rho1: float
            Density in neighboor cell (left)
        q0 : np.array(float), size: (2)
            Velocity in owner cell (right)
        q1 : np.array(float), size: (2)
            Velocity neighboor cell (left)
        h0: float
            Enthalpy in owner cell (right)
        h1: float
            Enthalpy in neighboor cell (left)

        Returns:
        roe_rho, roe_q, roe_c : tuple(float, np.array(float), size: (2), float)
            Roe's avergared density, velcoty and speed of sound
        """
        # Compute Roe's averaged variables
        z0 = np.sqrt(rho0)
        z1 = np.sqrt(rho1)
        roe_rho = z0 * z1
        roe_q = (z0 * q0 + z1 * q1) / (z0 + z1)
        roe_h = (z0 * h0 + z1 * h1) / (z0 + z1)
        roe_c = self._flu.eval_speed_sound_enthalpy(roe_h, roe_q)
        return roe_rho, roe_q, roe_c
