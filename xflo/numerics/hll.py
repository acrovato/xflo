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

class HLL(Flux):
    """Harten-Lax-van Leer flux formulation
    Riemann Solvers and Numerical Methods for Fluid Dynamics, E.F. Toro,
    Springer, 2009
    https://link.springer.com/book/10.1007/b79761
    """
    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        # Reconstruct left/right states (since the normal is inward wrt cell 0, 0 is right and 1 is left)
        s0 = s0 - ds0.dot(n) * 0.5 * d
        s1 = s1 + ds1.dot(n) * 0.5 * d

        # Compute left/right primitives, speed of sound and enthalpy
        rho0, q0, p0 = self._flu.eval_primitive(s0)
        rho1, q1, p1 = self._flu.eval_primitive(s1)
        c0 = self._flu.eval_speed_sound(rho0, p0)
        c1 = self._flu.eval_speed_sound(rho1, p1)
        h0 = self._flu.eval_enthalpy(rho0, q0, p0)
        h1 = self._flu.eval_enthalpy(rho1, q1, p1)

        # Compute left/right fluxes
        f0 = self._flu.compute_flux(rho0, q0, p0)
        f1 = self._flu.compute_flux(rho1, q1, p1)

        # Compute Roe's averaged variables
        z0 = np.sqrt(rho0)
        z1 = np.sqrt(rho1)
        roe_q = (z0 * q0 + z1 * q1) / (z0 + z1)
        roe_h = (z0 * h0 + z1 * h1) / (z0 + z1)
        roe_c = self._flu.eval_speed_sound_enthalpy(roe_h, roe_q)

        # Compute projected velocities and signal speeds
        qn0 = q0.dot(n)
        qn1 = q1.dot(n)
        a0 = max(qn0 + c0, roe_q.dot(n) + roe_c)
        a1 = min(qn1 - c1, roe_q.dot(n) - roe_c)

        # Compute HLL flux
        if a1 > 0.:
            return f1.dot(n) * l
        elif a0 < 0.:
            return f0.dot(n) * l
        else:
            return (((a0 * f1 - a1 * f0).dot(n) + a1 * a0 * (s0 - s1)) / (a0 - a1)) * l
