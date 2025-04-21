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

from .gradient import Gradient
import numpy as np

class FiniteVolume:
    """Finite volume discretization

    Attributes:
    problem : xflo.structure.problem.Problem
        Problem definition
    _flx : Flux object
        Flux formulation
    _ncells : int
        Number of cells
    _states : np.array(float)
        Conservative variables vector
    _residuals : np.array(float)
        Residuals vector
    _dt : np.array(float)
        Local time step divided by cell area
    """
    def __init__(self, problem, flux, gradient=None):
        # Data objetcs
        self.problem = problem
        self._flx = flux
        self._grd = gradient if gradient is not None else Gradient(problem)

        # Set sizes
        self._ncells = self.problem.mesh.get_ncells() # number of cells

        # Conservative varibales and residuals
        n_dofs = self._ncells * self.problem.get_nukn() # number of degrees of freedom
        self._states = np.zeros(n_dofs, dtype=float)
        self._residuals = np.zeros(n_dofs, dtype=float)
        self._dt = np.zeros(n_dofs, dtype=float)

    def get_states(self):
        """"Returns:
        states : np.array(float)
        """
        return self._states

    def get_residuals(self):
        """"Returns:
        residuals : np.array(float)
        """
        return self._residuals

    def initialize(self):
        """Set states, residuals and problem
        """
        # Set states from freestream
        state_inf = self.problem.get_freestream_state()
        for i_var in range(self.problem.get_nukn()):
            self._states[i_var::self.problem.get_nukn()] = state_inf[i_var]

        # Update residuals and problem
        self._compute_residuals()
        self.problem.update(self._states, self._residuals)

    def update(self, states):
        """Set states and update problem

        Parameters:
        states : np.array(float)
            Conservative variables vector
        """
        self._states = states
        self._compute_residuals()
        self.problem.update(self._states, self._residuals)

    def compute_timestep(self, cfl):
        """Compute local time step divided by cell area correspoding to given CFL number

        Parameters:
        cfl : float
            CFL number

        Returns:
        dt_a : np.array(float)
            Local time step of cells divided by cell area
        """
        area = self.problem.mesh.get_cells_area()
        u = self.problem.get_variables('VelocityX')
        v = self.problem.get_variables('VelocityY')
        c = self.problem.get_variables('SpeedOfSound')
        for i_cell in range(self._ncells):
            a = np.linalg.norm(np.array([u[i_cell], v[i_cell]])) + c[i_cell] # wave speed
            self._dt[self.problem.get_uids(i_cell)] = cfl * np.sqrt(area[i_cell]) / a / area[i_cell]
        return self._dt

    def _compute_jacobian(self):
        raise

    def _compute_residuals(self):
        """Compute cell residuals"""
        # Reset residuals
        self._residuals[:] = 0.

        # Compute gradients
        grads = self._grd.compute(self._states)

        # Get edge to cell connectivity
        cids = self.problem.mesh.get_edges_cells()

        # Compute fluxes on boundary edges and add residuals to cells
        for bc in self.problem.get_bcs():
            # get boundary edges and metrics
            bnd = self.problem.mesh.get_boundary(bc.get_name())
            eids = bnd.get_edge_ids()
            lgts, nrms = bnd.get_edge_metrics()
            dsts = bnd.get_cell_distances()
            for i_edge, eid in enumerate(eids):
                # neighboor cell and corresponding unknown indices
                cid = cids[eid][0]
                sid = self.problem.get_uids(cid)
                # integrated projected flux
                ghost_state = bc.compute_ghost(self._states[sid], nrms[i_edge])
                self._residuals[sid] += self._flx.compute_residual(self._states[sid], ghost_state, grads[sid], grads[sid], nrms[i_edge], lgts[i_edge], dsts[i_edge])

        # Compute flux on internal edges and add residuals to cells
        fld = self.problem.mesh.get_field()
        eids = fld.get_edge_ids()
        lgts, nrms = fld.get_edge_metrics()
        dsts = fld.get_cell_distances()
        for i_edge, eid in enumerate(eids):
            # "left" and "right" neighboor cells
            cid0 = cids[eid][0]
            cid1 = cids[eid][1]
            # corresponding unknown indices
            sid0 = self.problem.get_uids(cid0)
            sid1 = self.problem.get_uids(cid1)
            # integrated projected flux
            ipflx = self._flx.compute_residual(self._states[sid0], self._states[sid1], grads[sid0], grads[sid1], nrms[i_edge], lgts[i_edge], dsts[i_edge])
            self._residuals[sid0] += ipflx
            self._residuals[sid1] -= ipflx
