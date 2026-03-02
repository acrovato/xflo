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
            Conservative variables (density, density momentum and density stagnation energy)

        Returns:
        rho, q, p : tuple(float)
            Primitive variables (density, velocity and pressure)
        """
        rho = state[0]
        q = state[1:3] / rho
        p = (self._g - 1) * (state[3] - 0.5 * rho * q.dot(q))
        return rho, q, p

    def eval_state(self, primitive):
        """Evaluate conservative variables from primitive variables

        Parameters:
        primitive : np.array(float)
            Primitive variables (density, velocity and pressure)

        Returns:
        rho, rhoq, rhoe : np.array(float)
            Conservative variables (density, density momentum and density stagnation energy)
        """
        rho = primitive[0]
        rhoq = rho * np.array([primitive[1], primitive[2]])
        rhoe = primitive[3] / (self._g - 1) + 0.5 * rhoq.dot(rhoq) / rho # p = rho * (gamma-1) * (e0-q^2/2)
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

    def eval_speed_sound_enthalpy(self, h, q):
        """Evaluate speed of sound from enthalpy

        Parameters:
        h : float
            Specific stagnation enthalpy
        q : np.array(float)
            Velocity

        Returns:
        c : float
            Speed of sound
        """
        return np.sqrt((self._g - 1) * (h - 0.5 * q.dot(q)))

    def eval_enthalpy(self, rho, q, p):
        """Evaluate specific stagnation enthalpy

        Parameters:
        rho : float
            Density
        q : np.array(float)
            Velocity
        p : float
            Pressure

        Returns:
        h : float
            Specific stagnation enthalpy
        """
        return self._g / (self._g - 1) * p / rho + 0.5 * q.dot(q) # (rhoe + p) / rho

    def eval_entropy(self, rho, c):
        """Evaluate specific entropy

        Parameters:
        rho : float
            Density
        c : float
            Speed of sound

        Returns:
        s : float
            Specific entropy
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
            Specific entropy
        c : float
            Speed of sound
        q : np.array(float)
            Velocity

        Returns:
        rho, rhoe : tuple(float)
            Density and density stagnation energy
        """
        rho = ((c * c) / (self._g * s)) ** (1 / (self._g - 1)) # rho = (c^2 / (gamma*s)) ^ 1/(gamma-1)
        p = rho * c * c / self._g # p = rho * c^2 / gamma
        rhoe = p / (self._g - 1) + 0.5 * rho * q.dot(q) # p = (gamma-1) * rho * (e0 - 1/2 q^2)
        return rho, rhoe

    def compute_flux(self, rho, q, p):
        """Compute the Euler flux vector

        Parameters:
        rho : float
            Density
        q : np.array(float)
            Velocity vector
        p : float
            Pressure

        Returns:
        f : np.array(float)
            Flux vector
        """
        # Pre compute common factor
        rhoq = rho * q
        h = self.eval_enthalpy(rho, q, p)

        # Compute flux
        f = np.zeros((4, 2))
        f[0, :] = np.array([rhoq[0], rhoq[1]]) # [rho*u, rho*v]
        f[1, :] = np.array([rhoq[0] * q[0] + p, rhoq[1] * q[0]]) # [rho*u*u+p, rho*u*v]
        f[2, :] = np.array([rhoq[0] * q[1], rhoq[1] * q[1] + p]) # [rho*u*v, rho*v*v+p]
        f[3, :] = np.array([rhoq[0] * h, rhoq[1] * h]) # [(rho*e0+p)*u, (rho*e0+p)*v]
        return f

    def compute_eigen_decomposition(self, rho, q, c, n):
        """Compute eigenvalue and eigenvector decomposition of flux Jacobian matrix

        Parameters:
        rho : float
            Density
        q : np.array(float)
            Velocity vector
        c : float
            Speed of sound
        n : np.array(float)
            Edge unit normal vector

        Returns:
        lam, eig, ieig : tulple(np.array(float))
            Eigenvalue vector, eigenvector matrix, eigenvector inverse matrix
        """
        # Pre-compute factors
        sqvel = 0.5 * q.dot(q)
        qn = q.dot(n)
        rho_c = rho / c
        rhoc_g1 = (rho * c) / (self._g - 1)
        rho_n = rho * n
        g1_c2 = (self._g - 1) / (c * c)
        g1_rhoc = (self._g - 1) / (rho * c)
        n_rho = n / rho

        # Compute eignvalues
        lam = np.array([qn, qn, qn + c, qn - c])

        # Compute eigenvectors' matrix
        eig = np.zeros((4, 4))
        eig[0, :] = np.array([1.0,
                            0.0,
                            0.5 * rho_c,
                            0.5 * rho_c])
        eig[1, :] = np.array([q[0],
                            rho_n[1],
                            0.5 * (q[0] * rho_c + rho_n[0]),
                            0.5 * (q[0] * rho_c - rho_n[0])])
        eig[2, :] = np.array([q[1],
                            -rho_n[0],
                            0.5 * (q[1] * rho_c + rho_n[1]),
                            0.5 * (q[1] * rho_c - rho_n[1])])
        eig[3, :] = np.array([sqvel,
                            rho_n[1] * q[0] - rho_n[0] * q[1],
                            0.5 * (sqvel * rho_c + rho_n[0] * q[0] + rho_n[1] * q[1] + rhoc_g1),
                            0.5 * (sqvel * rho_c - rho_n[0] * q[0] - rho_n[1] * q[1] + rhoc_g1)])

        # Compute inverse eigenvectors' matrix
        ieig = np.zeros((4, 4))
        ieig[0, :] = np.array([1.0 - g1_c2 * sqvel,
                            g1_c2 * q[0],
                            g1_c2 * q[1],
                            -g1_c2])
        ieig[1, :] = np.array([-n_rho[1] * q[0] + n_rho[0] * q[1],
                            n_rho[1],
                            -n_rho[0],
                            0.0])
        ieig[2, :] = np.array([-n_rho[0] * q[0] - n_rho[1] * q[1] + g1_rhoc * sqvel,
                            n_rho[0] - g1_rhoc * q[0],
                            n_rho[1] - g1_rhoc * q[1],
                            g1_rhoc])
        ieig[3, :] = np.array([n_rho[0] * q[0] + n_rho[1] * q[1] + g1_rhoc * sqvel,
                            -n_rho[0] - g1_rhoc * q[0],
                            -n_rho[1] - g1_rhoc * q[1],
                            g1_rhoc])
        return lam, eig, ieig
