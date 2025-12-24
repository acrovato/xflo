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

class Roe(Flux):
    """Roe flux formulation
    TODO ref
    TODO limiter + MUSCL?

    Attributes:
    _efix : float
        Entropy fix coefficient
    """
    def __init__(self, fluid, entropy_fix=1e-6):
        self._efix = entropy_fix
        super().__init__(fluid)

    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        # Compute left/right primitives
        rho0, q0, p0 = self._flu.eval_primitive(s0)
        rho1, q1, p1 = self._flu.eval_primitive(s1)

        # Compute left/right fluxes
        f0 = self._flu.compute_flux(rho0, q0, p0, s0[3])
        f1 = self._flu.compute_flux(rho1, q1, p1, s1[3])
        f = 0.5 * (f0 + f1)

        # Compute Roe-averaged variables
        z0 = np.sqrt(rho0)
        z1 = np.sqrt(rho1)
        roe_rho = z0 * z1
        roe_q = (z0 * q0 + z1 * q1) / (z0 + z1)
        roe_p = (z0 * p0 + z1 * p1) / (z0 + z1) # average pressure instead of enthalpy
        roe_c = self._flu.eval_speed_sound(roe_rho, roe_p)

        # Compute eigenvalues and eigenvectors
        lam, eig_mat, eig_imat = self._flu.compute_eigen_decomposition(roe_rho, roe_q, roe_c, n)
        lam = np.maximum(np.abs(lam), self._efix * max(np.abs(lam))) # fix eigenvalues to avoid expansion shocks

        # Compute ROE flux
        jac = eig_mat @ np.diag(lam) @ eig_imat
        return (f.dot(n) - 0.5 * jac.dot(s0 - s1)) * l
