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

from .riemann import Riemann

class HLL(Riemann):
    """Harten-Lax-van Leer flux formulation
    Riemann Solvers and Numerical Methods for Fluid Dynamics, E.F. Toro,
    Springer, 2009
    https://link.springer.com/book/10.1007/b79761
    """
    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        # Reconstruct left/right states
        s0, s1 = self._reconstruct_states(s0, s1, ds0, ds1, n, d)

        # Compute left/right primitives, enthalpy and fluxes
        rho0, q0, _, c0, h0, f0 = self._eval_vars_flux(s0)
        rho1, q1, _, c1, h1, f1 = self._eval_vars_flux(s1)

        # Compute Roe's averaged variables
        _, roe_q, roe_c = self._eval_roe_average(rho0, rho1, q0, q1, h0, h1)

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
