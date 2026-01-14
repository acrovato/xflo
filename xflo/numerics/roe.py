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
import numpy as np

class Roe(Riemann):
    """Roe flux formulation
    Riemann Solvers and Numerical Methods for Fluid Dynamics, E.F. Toro,
    Springer, 2009
    https://link.springer.com/book/10.1007/b79761

    Attributes:
    _efix : float
        Entropy fix coefficient
    """
    def __init__(self, fluid, entropy_fix=1e-6):
        self._efix = entropy_fix
        super().__init__(fluid)

    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        # Reconstruct left/right states
        s0, s1 = self._reconstruct_states(s0, s1, ds0, ds1, n, d)

        # Compute left/right primitives, enthalpy and fluxes
        rho0, q0, _, _, h0, f0 = self._eval_vars_flux(s0)
        rho1, q1, _, _, h1, f1 = self._eval_vars_flux(s1)

        # Compute Roe's averaged variables
        roe_rho, roe_q, roe_c = self._eval_roe_average(rho0, rho1, q0, q1, h0, h1)

        # Compute eigenvalues and eigenvectors
        lam, eig_mat, eig_imat = self._flu.compute_eigen_decomposition(roe_rho, roe_q, roe_c, n)
        lam = np.maximum(np.abs(lam), self._efix * max(np.abs(lam))) # fix eigenvalues to avoid expansion shocks

        # Compute Roe's flux
        jac = eig_mat @ np.diag(lam) @ eig_imat
        return 0.5 * ((f0 + f1).dot(n) - jac.dot(s0 - s1)) * l
