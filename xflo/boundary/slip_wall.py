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

from .boundary_condition import BoundaryCondition
import numpy as np

class SlipWall(BoundaryCondition):
    """Slip wall boundary condition
    Preserve tangential velocity and mirror normal velocity
    """
    def __init__(self, fluid, boundary_name):
        super().__init__(fluid, boundary_name)

    def compute_ghost(self, state, n):
        # Get momentum and reflect normal component
        rq = state[1:3]
        rq_reflected = rq - 2 * rq.dot(n) * n
        return np.array([state[0], rq_reflected[0], rq_reflected[1], state[3]])
