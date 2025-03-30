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

from xflo.utils.error import XFloNotImplemented

class BoundaryCondition:
    """Boundary conditions

    Attributes:
    _flu : Fluid object
        Constitutive law relations
    _name : str
        Name of the physical boundary associated to this boundary condition
    """
    def __init__(self, fluid, boundary_name):
        self._flu = fluid
        self._name = boundary_name

    def get_name(self):
        """Returns:
        boundary name : str
        """
        return self._name

    def compute_ghost(self, state, n):
        """Compute conservative variables at boundary edge

        Parameters:
        state : np.array(4)
            Conservatives variables in the interior cell linked to the edge
        n : np.array(2)
            Edge unit normal vector

        Returns:
        ghost_state : np.array(4)
            Conservatives variables at boundary edge
        """
        raise XFloNotImplemented('Boundary condition not implemented!')
