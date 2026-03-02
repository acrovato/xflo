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

from xflo.utils.gradient import compute_grad_fd
from xflo.utils.error import XFloNotImplemented

class Flux:
    """Flux formulation

    Attributes:
    _flu : Fluid object
        Constitutive relations
    """
    def __init__(self, fluid):
        self._flu = fluid

    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        """Compute and integrate the flux projected on the edge normal

        Parameters:
        s0 : np.array(float), size: (4)
            Conservatives variables in the edge's owner cell
        s1 : np.array(float), size: (4)
            Conservatives variables in the edge's neighboor cell
        ds0 : np.array(float), size: (4, 2)
            Gradient of conservatives variables in the edge's owner cell
        ds1 : np.array(float), size: (4, 2)
            Gradient of conservatives variables in the edge's neighboor cell
        n : np.array(float), size: (2)
            Edge unit normal vector
        l : float
            Edge length
        d : float
            Distance between owner and neighboor cells linked to edge

        Returns:
        flux_proj : np.array(float), size: (4, 1)
            Integrated flux projected on the edge normal direction
        """
        raise XFloNotImplemented('Flux not implemented!')

    def compute_jacobian(self, s0, s1, ds0, ds1, n, l, d):
        """Compute gradient of flux with respect to conservative variables

        Parameters:
        s0 : np.array(float), size: (4)
            Conservatives variables in the edge's owner cell
        s1 : np.array(float), size: (4)
            Conservatives variables in the edge's neighboor cell
        ds0 : np.array(float), size: (4, 2)
            Gradient of conservatives variables in the edge's owner cell
        ds1 : np.array(float), size: (4, 2)
            Gradient of conservatives variables in the edge's neighboor cell
        n : np.array(float), size: (2)
            Edge unit normal vector
        l : float
            Edge length
        d : float
            Distance between owner and neighboor cells linked to edge

        Returns:
        Gradient of flux wrt. conservatives variables : list(np.array(float), size: (4, 4)), size: (2)
        """
        args = [s0, s1, ds0, ds1, n, l, d]
        wrt = [0, 1]
        return compute_grad_fd(self.compute_residual, args, wrt)
