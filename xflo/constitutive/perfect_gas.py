# -*- coding: utf-8 -*-

# xflo
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

class PerfectGas:
    """Perfect gas constitutive law equations
    TODO
    """
    def __init__(self, gamma=1.4):
        self._g = gamma

    def eval_primitive(self, state):
        """TODO"""
        rho = state[0]
        q = state[1:3] / rho
        p = (self._g - 1) * (state[3] - 0.5 * rho * q.dot(q))
        return rho, q, p

    def eval_state(self, primitive):
        """TODO"""
        rho = primitive[0]
        rhoq = rho * np.array([primitive[1], primitive[2]])
        rhoe = primitive[3] / (self._g - 1) + 0.5 * rhoq.dot(rhoq) / rho # p = rho * (g - 1) * (e - u^2/2)
        return np.array([rho, rhoq[0], rhoq[1], rhoe])

    def compute_speed_sound(self, rho, p):
        """TODO"""
        c = np.sqrt(self._g * p / rho) # c^2 = gamma*R*T
        return c

    def compute_entropy(self, rho, c):
        """TODO"""
        return c * c / (self._g * rho ** (self._g - 1))

    def compute_riemann(self, u, c, sign):
        """TODO"""
        return u + sign * 2 * c / (self._g - 1)

    def compute_speed_riemann(self, rp, rm):
        """TODO
        """
        u = 0.5 * (rp + rm)
        c = 0.25 * (self._g - 1) * (rp - rm)
        return u, c

    def compute_states_entropy(self, s, c, q):
        """TODO"""
        rho = ((c * c) / (self._g * s)) ** (1 / (self._g - 1))
        p = rho * c * c / self._g
        rhoe = p / (self._g - 1) + 0.5 * rho * q.dot(q)
        return rho, rhoe

    def compute_flux(self, rho, q, p, rhoe):
        """Compute the Euler flux vector

        Arguments:
        state : np.array(4)
            Vector of conserved variables

        Returns:
        flux : np.array(4, 2)
            Flux vector
        """
        # Get velocity components
        u = q[0]
        v = q[1]

        # Compute flux
        f = np.zeros((4, 2))
        f[0, :] = np.array([rho * u, rho * v]) # [rho*u, rho*v]
        f[1, :] = np.array([rho * u * u + p, rho * u * v]) # [rho*u*u+p, rho*u*v]
        f[2, :] = np.array([rho * u * v, rho * v * v + p]) # [rho*u*v, rho*v*v+p]
        f[3, :] = np.array([(rhoe + p) * u, (rhoe + p) * v]) # [(rho*E+p)*u, (rho*E+p)*v]
        return f
