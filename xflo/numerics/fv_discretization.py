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

import numpy as np

class FiniteVolume:
    """Finite volume discretization

    Attributes:
    _problem : Problem object
        Problem definition
    _flux : Flux object
        Flux formulation
    """
    def __init__(self, problem, flux):
        # Data objetcs
        self.problem = problem
        self._flx = flux

        # Set sizes
        self._nvar_cell = 4 # number of variables per cell
        self._ncells = self.problem.mesh.get_ncells() # number of cells

        # Conservative varibales and residuals
        n_dofs = self._ncells * self._nvar_cell # number of degrees of freedom
        self._states = np.zeros(n_dofs, dtype=float)
        self._residuals = np.zeros(n_dofs, dtype=float)

    def get_states(self):
        """"TODO"""
        return self._states

    def get_residuals(self):
        return self._residuals

    def set_states(self, states):
        """Set states and update primitives"""
        self._states = states
        for i_cell in range(self._ncells):
            self.problem.update_variable(i_cell, self._states[self._get_uids(i_cell)])

    def initialize(self):
        """Set initial conditions
        """
        # Get primitives
        rinf, uinf, vinf, pinf, _ = self.problem.get_variables()
        state_inf = self.problem.fluid.eval_state(np.array([rinf[0], uinf[0], vinf[0], pinf[0]]))
        for i_var in range(self._nvar_cell):
            self._states[i_var::self._nvar_cell] = state_inf[i_var]

    def compute_jacobian(self):
        raise

    def compute_residuals(self):
        """Compute cell residuals"""
        # Reset residuals
        self._residuals[:] = 0.

        # Compute fluxes on boundary edges and add residuals to cells
        for bc in self.problem.get_bcs():
            cids, lgts, nrms = self.problem.mesh.get_boundary_edges(bc.get_name())
            for i_edge in range(cids.shape[0]):
                # neighboor cell and corresponding unknown indices
                cid = cids[i_edge][0]
                sid = self._get_uids(cid)
                # integrated projected flux
                ghost_state = bc.compute_ghost(self._states[sid], nrms[i_edge])
                self._residuals[sid] += self._flx.compute_residual(self._states[sid], ghost_state, nrms[i_edge], lgts[i_edge])

        # Compute flux on internal edges and add residuals to cells
        cids, lgts, nrms = self.problem.mesh.get_field_edges()
        for i_edge in range(cids.shape[0]):
            # "left" and "right" neighboor cells
            cid0 = cids[i_edge][0]
            cid1 = cids[i_edge][1]
            # corresponding unknown indices
            sid0 = self._get_uids(cid0)
            sid1 = self._get_uids(cid1)
            # integrated projected flux
            ipflx = self._flx.compute_residual(self._states[sid0], self._states[sid1], nrms[i_edge], lgts[i_edge])
            self._residuals[sid0] += ipflx
            self._residuals[sid1] -= ipflx

    def _get_uids(self, cid):
        """Get unkown indices corresponding to cell
        TODO"""
        return np.array(range(cid * self._nvar_cell, (cid + 1) * self._nvar_cell))
