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

class BoundaryCondition:
    """Boundary conditions

    Attributes:
    _flu : Fluid object
        Constitutive law relations
    _bnd_name : str
        Name of the physical boundary associated to this boundary condition
    """
    def __init__(self, fluid, boundary_name):
        self._flu = fluid
        self._bnd_name = boundary_name

    def get_name(self):
        """Returns:
        boundary name : str
        """
        return self._bnd_name

    def compute_ghost(self, state, n):
        """Compute conservative variables at boundary edge

        Arguments:
        state : np.array(4)
            Conservatives variables in the interior cell linked to the edge
        n : np.array(2)
            Edge unit normal vector

        Returns:
        ghost_state : np.array(4)
            Conservatives variables at boundary edge
        """
        raise NotImplementedError('Boundary condition not implemented!')
