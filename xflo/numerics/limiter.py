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

class Limiter:
    """Method to compute cell limiters

    Attributes:
    _pbl : xflo.structure.problem.Problem
        Problem definition
    _lim : np.array(float)
        Limiter on conservative variables vector
    """
    def __init__(self, problem):
        self._pbl = problem
        self._lim = np.ones(problem.mesh.get_ncells() * problem.get_nukn(), dtype=float)

    def compute(self, states, gradients):
        """Compute the limiters at the cells

        Parameters:
        states : np.array(float)
            Conservative variables vector
        gradients : np.array(float)
            Gradient of conservative variables vector

        Returns:
        _lim : np.array(float)
            Limiter on conservative variables vector
        """
        return self._lim
