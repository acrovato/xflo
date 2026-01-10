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

class ExplicitRungeKutta(TimeIntegration):
    """Explicit generic Runge Kutta time integration method

    U(n+1) = U(n) + dt / A * sum_i( b_i * k_i )
    k_i = R( sum_j( U(n) + dt / A * a_ij * k_j ) )

    Attributes:
    _a : array(array(float))
        Runge-Kutta matrix (a_ij coefficients in Butcher's tableau)
    _b = array(float)
        Runge-Kutta weights (b_i coefficients in Butcher's tableau)
    _nstages : int
        Number of stages
    """
    def __init__(self, discretization, writer, init_cfl=0.1, rel_tol=1e-6, max_iter=1000, save_freq=100):
        super().__init__(discretization, writer, init_cfl, 0., rel_tol, max_iter, save_freq)
        # Set default values to explicit Euler
        self.set_coeff()

    def set_coeff(self, a=None, b=[1]):
        """Set Butcher's tableau coefficients for Runge-Kutta

        Parameters:
        a : array(array(float))
            Runge-Kutta matrix (a_ij coefficients in Butcher's tableau)
        b = array(float)
            Runge-Kutta weights (b_i coefficients in Butcher's tableau)
        """
        self._a = a
        self._b = b
        self._nstages = len(b)

    def _update_solution(self):
        # Get solution and residuals, and compute time step
        s0 = self._disc.get_states()
        r = [None] * self._nstages
        r[0] = self._disc.get_residuals()
        dt = self._disc.compute_timestep(self._cfl)

        # Compute stages
        for i in range(1, self._nstages):
            s = s0.copy()
            for j in range(i):
                s += dt * self._a[i-1][j] * r[j]
            self._disc.update(s)
            r[i] = self._disc.get_residuals()

        # Update solution
        s = s0.copy()
        for i in range(self._nstages):
            s += dt * self._b[i] * r[i]
        self._disc.update(s)

class ExplicitEuler(ExplicitRungeKutta):
    """Explicit (forward) Euler time integration method

    U(n+1) = U(n) + dt / A * R(n)
    """
    def __init__(self, discretization, writer, init_cfl=0.1, rel_tol=1e-6, max_iter=1000, save_freq=100):
        super().__init__(discretization, writer, init_cfl, rel_tol, max_iter, save_freq)

class ExplicitRk4(ExplicitRungeKutta):
    """Explicit 4-stages order 4 Runge Kutta time integration method

    U(n+1) = U(n) + dt / A * sum_i( b_i * k_i )
    k_i = R( sum_j( U(n) + dt / A * a_ij * k_j ) )
    with
    a = [[1/2],
         [0, 1/2],
         [0, 0, 1]]
    b = [1/6, 1/3, 1/3, 1/6]
    """
    def __init__(self, discretization, writer, init_cfl=0.1, rel_tol=1e-6, max_iter=1000, save_freq=100):
        super().__init__(discretization, writer, init_cfl, rel_tol, max_iter, save_freq)
        self.set_coeff([[1. / 2.], [0., 1. / 2.], [0., 0., 1.]], [1./.6, 1./3., 1./3., 1./6.])
