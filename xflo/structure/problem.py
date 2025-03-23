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

from xflo.boundary.riemann_invariant import RiemannInvariant
from xflo.boundary.slip_wall import SlipWall
from xflo.structure.body import Body
import numpy as np

class Problem:
    """TODO

    Attributes:
    Mesh, constitutive law, boundary condtions and bodies of interest
    """
    def __init__(self, mesh, fluid):
        # Data objects
        self.mesh = mesh
        self.fluid = fluid
        self._far = None
        self._wal = []
        self._bdy = []

        # Default freestream
        self._aoa = 0.
        self._minf = 0.3
        self._rinf = 1.225
        self._pinf = 101325

        # Default geometry
        self._c = 1.
        self._xc = np.array([0.25, 0.])

        # Solution
        n = self.mesh.get_ncells()
        self._rho = np.zeros(n, dtype=float)
        self._u = np.zeros(n, dtype=float)
        self._v = np.zeros(n, dtype=float)
        self._p = np.zeros(n, dtype=float)
        self._mach = np.zeros(n, dtype=float)

    def get_bcs(self):
        """Get boundary conditions
        TODO"""
        return [self._far, *self._wal]

    def get_variables(self):
        """Get flow variables
        TODO"""
        return self._rho, self._u, self._v, self._p, self._mach

    def get_solution(self):
        """TODO"""
        return {'rho': self._rho, 'u': self._u, 'v': self._v, 'p': self._p, 'mach': self._mach}

    def set_farfield(self, name):
        """Create farfield boundary condition
        """
        self._far = RiemannInvariant(self.fluid, name)

    def set_wall(self, name):
        """Create wall boundary condition and monitor it
        """
        self._wal.append(SlipWall(self.fluid, name))
        #self._bdy.append(Body(name))

    def set_freestream(self, aoa, mach, rho, p):
        """TODO"""
        # Set freestream state
        self._aoa = aoa * np.pi / 180.
        self._minf = mach
        self._rinf = rho
        self._pinf = p
        self._far.set_freestream(self._aoa, mach, rho, p)

        # Initialize primitives
        cinf = self.fluid.compute_speed_sound(rho, p)
        self._rho[:] = rho
        self._u[:] = mach * cinf * np.cos(self._aoa)
        self._v[:] = mach * cinf * np.sin(self._aoa)
        self._p[:] = p
        self._mach[:] = mach

    def set_geometry(self, c_ref, x_ref, z_ref):
        """TODO"""
        self._c = c_ref
        self._xc[0] = x_ref
        self._xc[1] = z_ref

    def update_variable(self, i_cell, state):
        """TODO"""
        self._rho[i_cell], q, self._p[i_cell] = self.fluid.eval_primitive(state)
        self._u[i_cell] = q[0]
        self._v[i_cell] = q[1]
        self._mach[i_cell] = np.linalg.norm(q) / self.fluid.compute_speed_sound(self._rho[i_cell], self._p[i_cell])
