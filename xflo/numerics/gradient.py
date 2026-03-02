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

from .limiter import Limiter
from xflo.utils.error import XFloNotImplemented
import numpy as np

class Gradient:
    """Method to compute the gradients

    Attributes:
    _pbl : xflo.structure.problem.Problem
        Problem definition
    _lim : xflo.numerics.limiter.Limiter
        Limiter calculation
    _states : np.array(float)
        Conservative variables vector at edges
    _grads : np.array(float)
        Gradient of conservative variables vector
    """
    def __init__(self, problem, limiter=None):
        self._pbl = problem
        self._lim = limiter if limiter is not None else Limiter(problem)
        self._states = np.zeros(problem.mesh.get_nedges() * problem.get_nukn(), dtype=float)
        self._grads = np.zeros((problem.mesh.get_ncells() * problem.get_nukn(), 2), dtype=float)

    def compute(self, states):
        """Compute limited gradients at cells centers

        Parameters:
        states : np.array(float)
            Conservative variables vector

        Returns:
        _grads : np.array(float)
            Gradient of conservative variables vector
        """
        self._compute_gradients(states)
        self._grads *= self._lim.compute(states, self._grads)[:, np.newaxis]
        return self._grads

    def _compute_flux(self, state, n, l):
        """Compute integrated flux of states on edge

        Parameters:
        state : np.array(float)
            Conservative variables vector at edge
        n : np.array(float)
            Edge unit normal vector
        l : float
            Edge length

        Returns:
        flux : np.array(float)
            Integral of edge state multiplied by the normal vector
        """
        return state[:, np.newaxis] * np.tile(n, (self._pbl.get_nukn(), 1)) * l

    def _compute_gradients(self, states):
        """Actual implementation to compute gradients

        Parameters:
        states : np.array(float)
            Conservative variables vector
        """
        pass
