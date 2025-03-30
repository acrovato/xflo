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

class Flux:
    """Flux formulation

    Attributes:
    _flu : Fluid object
        Constitutive relations
    """
    def __init__(self, fluid):
        self._flu = fluid

    def compute_residual(self, state0, state1, n, l):
        """Compute and integrate the flux projected on the edge normal

        Parameters:
        state0 : np.array(4)
            Conservatives variables in the first cell linked to the face
        state1 : np.array(4)
            Conservatives variables in the second cell linked to the face
        n : np.array(2)
            Edge unit normal vector
        l : float
            Edge length

        Returns:
        flux_proj : np.array(4)
            Numerical flux projected on the face normal direction
        """
        raise XFloNotImplemented('Flux not implemented!')
