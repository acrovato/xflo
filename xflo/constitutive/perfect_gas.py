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

class PerfectGas:
    """Perfect gas constitutive law equations

    Attributes:
    _g : float
        Heat capacity ratio
    """
    def __init__(self, gamma=1.4):
        self._g = gamma

    def eval_primitive(self, state):
        """Evaluate primitive variables from conservative variables

        Parameters:
        state : np.array(float)
            Conservative variables
        Returns:
        rho, q, p : tuple(float)
            Primitive variables
        """
        rho = state[0]
        q = state[1:3] / rho
        p = (self._g - 1) * (state[3] - 0.5 * rho * q.dot(q))
        return rho, q, p

    def eval_state(self, primitive):
        """Evaluate conservative variables from primitive variables

        Parameters:
        primitive : np.array(float)
            Primitive variables
        Returns:
        rho, rhoq, rhoe : np.array(float)
            Conservative variables
        """
        rho = primitive[0]
        rhoq = rho * np.array([primitive[1], primitive[2]])
        rhoe = primitive[3] / (self._g - 1) + 0.5 * rhoq.dot(rhoq) / rho # p = rho * (g - 1) * (e - u^2/2)
        return np.array([rho, rhoq[0], rhoq[1], rhoe])

    def eval_speed_sound(self, rho, p):
        """Evaluate speed of sound

        Parameters:
        rho : float
            Density
        p : float
            Pressure

        Returns:
        c : float
            Speed of sound
        """
        return np.sqrt(self._g * p / rho) # c^2 = gamma*R*T

    def eval_entropy(self, rho, c):
        """Evaluate entropy

        Parameters:
        rho : float
            Density
        c : float
            Speed of sound

        Returns:
        s : float
            Entropy
        """
        return c * c / (self._g * rho ** (self._g - 1)) # s = c^2 / (gamma * rho^(gamma-1))

    def eval_riemann(self, u, c, sign):
        """Evaluate Riemann invariant

        Parameters:
        u : float
            Normal velocity
        c : float
            Speed of sound
        sign : float (+/- 1)
            Sign of the invariant

        Returns:
        r : float
            Riemann invariant
        """
        return u + sign * 2 * c / (self._g - 1) # R(+-) = u_n +- 2*c / (gamma-1)

    def eval_speed_riemann(self, rp, rm):
        """Evaluate propagation speeds based on Riemann invariant

        Parameters:
        rp : float
            Outgoing Riemann invariant
        rm : float
            Incoming Riemann invariant

        Returns:
        u, c : tuple(float)
            Normal velocity and speed of sound
        """
        u = 0.5 * (rp + rm)
        c = 0.25 * (self._g - 1) * (rp - rm)
        return u, c

    def eval_states_entropy(self, s, c, q):
        """Evaluate conservative variables from entropy

        Parameters:
        s : float
            Entropy
        c : float
            Speed of sound
        q : np.array(float)
            Velocity

        Returns:
        rho, rhoe : tuple(float)
            Density and total energy
        """
        rho = ((c * c) / (self._g * s)) ** (1 / (self._g - 1)) # rho = (c^2 / (gamma*s)) ^ 1/(gamma-1)
        p = rho * c * c / self._g # p = rho * c^2 / gamma
        rhoe = p / (self._g - 1) + 0.5 * rho * q.dot(q) # p = (gamma-1) * rho * (E - 1/2 q^2)
        return rho, rhoe

    def compute_flux(self, rho, q, p, rhoe):
        """Compute the Euler flux vector

        Parameters:
        rho : float
            Density
        q : np.array(float)
            Velocity vector
        p : float
            Pressure
        rhoe : float
            Total energy

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
