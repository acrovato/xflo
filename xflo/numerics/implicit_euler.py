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

from .time_integration import TimeIntegration
from xflo.utils.log import logger
import numpy as np
from scipy.sparse.linalg import gmres, spilu, LinearOperator

class ImplicitEuler(TimeIntegration):
    """Implicit (backward) Euler time integration method

    (J(n) - I * A/dt) * dU(n) = -R(n)

    Attributes:
    _irtol : float
        Relative tolerance for inner GMRES solver
    _iatol : float
        Absolute tolerance for inner GMRES solver
    _imxit : int
        Maximum number of iterations for inner GMRES solver
    _rstrt : int
        Number of iterations between restarts for inner GMRES solver
    _pdrop : float
        Drop tolerance for ILU preconditioner
    _pfill : float
        Fill ratio upper bound for ILU preconditioner
    """
    def __init__(self, discretization, writer, init_cfl=1., exp_cfl=0.7, rel_tol=1e-6, max_iter=100, save_freq=10):
        super().__init__(discretization, writer, init_cfl, exp_cfl, rel_tol, max_iter, save_freq)
        # Set default values
        self.set_solver_parameters()
        self.set_preconditioner_parameters()

    def set_solver_parameters(self, rel_tol=1e-3, abs_tol=1e-5, max_iter=10, n_restart=20):
        """Set inner GMRES solver parameters

        Parameters:
        rel_tol : float
            Relative tolerance
        abs_tol : float
            Absolute tolerance
        max_iter : int
            Maximum number of iterations
        n_restart : int
            Number of iterations between restarts
        """
        self._irtol = rel_tol
        self._iatol = abs_tol
        self._imxit = max_iter
        self._rstrt = n_restart

    def set_preconditioner_parameters(self, drop_tol=1e-6, fill_factor=20.):
        """Set ILU preconditioner parameters

        Parameters:
        drop_tol : float
            Drop tolerance for ILU preconditioner
        fill_factor : float
            Fill ratio upper bound for ILU preconditioner
        """
        self._pdrop = drop_tol
        self._pfill = fill_factor

    def _update_solution(self):
        # Get states and residuals
        s = self._disc.get_states()
        r = self._disc.get_residuals()

        # Get Jacobian and add A/dt on the diagonal
        j = self._disc.get_jacobian()
        n = s.shape[0]
        j.setdiag(j.diagonal() - np.ones(n) / self._disc.compute_timestep(self._cfl))

        # Build preconditioner
        p = spilu(j, drop_tol=self._pdrop, fill_factor=self._pfill)
        m = LinearOperator((n, n), p.solve)

        # Solve linear set of equations
        ds, info = gmres(j, -r, x0=s, rtol=self._irtol, atol=self._iatol, restart=self._rstrt, maxiter=self._imxit, M=m)
        if info > 0:
            logger.warning('Maximum number of iterations exceeded in GMRES solver.')

        # Update solution
        self._disc.update(s + ds, update_jac=True)
