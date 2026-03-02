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
import scipy.sparse as sp

class FiniteVolume:
    """Finite volume discretization

    Attributes:
    problem : xflo.structure.problem.Problem
        Problem definition
    _flx : xflo.numerics.flux.Flux
        Flux formulation
    _grd : xflo.numerics.gradient.Gradient
        Gradient calculation
    _ncells : int
        Number of cells
    _states : np.array(float)
        Conservative variables vector
    _residuals : np.array(float)
        Residuals vector
    _jacobian : sp.csc_matrix(float)
        Jacobian matrix
    _dt : np.array(float)
        Local time step divided by cell area
    """
    def __init__(self, problem, flux, gradient=None):
        # Data objetcs
        self.problem = problem
        self._flx = flux
        self._grd = gradient if gradient is not None else Gradient(problem)

        # Get number of cells
        self._ncells = self.problem.mesh.get_ncells()

        # Conservative varibales, Jacobian, residuals and timestep/area ratio
        n_dofs = self._ncells * self.problem.get_nukn() # number of degrees of freedom
        self._states = np.zeros(n_dofs, dtype=float)
        self._residuals = np.zeros(n_dofs, dtype=float)
        self._jacobian = sp.csc_matrix((n_dofs, n_dofs), dtype=float)
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

    def get_jacobian(self):
        """"Returns:
        jacobian : sp.csr_matrix(float)
        """
        return self._jacobian

    def initialize(self):
        """Set states, residuals and problem
        """
        # Set states from freestream
        sinf = self.problem.get_freestream_state()
        state_inf = np.tile(sinf, self._ncells)

        # Update residuals and problem
        self.update(state_inf, update_jac=True)

    def update(self, states, update_jac=False):
        """Set states and update problem

        Parameters:
        states : np.array(float)
            Conservative variables vector
        update_jac : bool
            True if Jacobian matrix is to be updated
        """
        # Update conservative variables, Jacobian, residuals and problem
        self._states = states
        if update_jac: self._compute_jacobian()
        self._compute_residuals()
        self.problem.update(self._states, self._residuals)

    def compute_timestep(self, cfl):
        """Compute local time step divided by cell area corresponding to given CFL number

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
        """Compute Jacobian matrix"""
        # Reset Jacobian
        rows = []
        cols = []
        data = []
        for i_col in range(self._jacobian.shape[1]):
            self._jacobian.data[self._jacobian.indptr[i_col]:self._jacobian.indptr[i_col + 1]] = 0.

        # Compute gradients (Jacobian is approximated because sensitivity of gradients are neglected)
        grads = self._grd.compute(self._states)

        # Get edge to cell connectivity
        cids = self.problem.mesh.get_edges_cells()

        # Compute fluxes sensitivity on boundary edges and build cells Jacobian
        for bc in self.problem.get_bcs():
            # get boundary edges and metrics
            bnd = self.problem.mesh.get_boundary(bc.get_name())
            eids = bnd.get_edge_ids()
            lgts, nrms = bnd.get_edge_metrics()
            dsts = bnd.get_cell_distances()
            for i_edge, eid in enumerate(eids):
                # neighboor cell unknown indices and reference ghost state
                sid = self.problem.get_uids(cids[eid][0])
                ghost_state = bc.compute_ghost(self._states[sid], nrms[i_edge])
                # flux Jacobian
                dg_ds = bc.compute_ghost_jacbian(self._states[sid], nrms[i_edge]) # gradient of ghost states wrt. boundary states
                f_s = self._flx.compute_jacobian(self._states[sid], ghost_state, grads[sid], grads[sid], nrms[i_edge], lgts[i_edge], dsts[i_edge]) # gradient of residuals wrt. boundary states
                jac = f_s[0] + f_s[1] @ dg_ds
                for i, i_row in enumerate(sid):
                    for j, j_row in enumerate(sid):
                        rows.append(i_row)
                        cols.append(j_row)
                        data.append(jac[i, j])

        # Compute flux sensitivity on internal edges and build cells Jacobian
        fld = self.problem.mesh.get_field()
        eids = fld.get_edge_ids()
        lgts, nrms = fld.get_edge_metrics()
        dsts = fld.get_cell_distances()
        for i_edge, eid in enumerate(eids):
            # owner and neighboor cells unknown indices
            sids = [self.problem.get_uids(cids[eid][0]), self.problem.get_uids(cids[eid][1])]
            # flux Jacobian
            jacs = self._flx.compute_jacobian(self._states[sids[0]], self._states[sids[1]], grads[sids[0]], grads[sids[1]], nrms[i_edge], lgts[i_edge], dsts[i_edge])
            for l, l_sgn in enumerate([+1, -1]):
                for i, i_row in enumerate(sids[l]):
                    for k, k_sid in enumerate(sids):
                        for j, j_row in enumerate(k_sid):
                            rows.append(i_row)
                            cols.append(j_row)
                            data.append(l_sgn * jacs[k][i, j])

        # Assemble Jacobian
        self._jacobian = sp.csc_matrix((data, (rows, cols)))

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
                # neighboor cell unknown indices
                sid = self.problem.get_uids(cids[eid][0])
                # integrated projected flux
                ghost_state = bc.compute_ghost(self._states[sid], nrms[i_edge])
                self._residuals[sid] += self._flx.compute_residual(self._states[sid], ghost_state, grads[sid], grads[sid], nrms[i_edge], lgts[i_edge], dsts[i_edge])

        # Compute flux on internal edges and add residuals to cells
        fld = self.problem.mesh.get_field()
        eids = fld.get_edge_ids()
        lgts, nrms = fld.get_edge_metrics()
        dsts = fld.get_cell_distances()
        for i_edge, eid in enumerate(eids):
            # owner and neighboor cells unknown indices
            sid0 = self.problem.get_uids(cids[eid][0])
            sid1 = self.problem.get_uids(cids[eid][1])
            # integrated projected flux
            ipflx = self._flx.compute_residual(self._states[sid0], self._states[sid1], grads[sid0], grads[sid1], nrms[i_edge], lgts[i_edge], dsts[i_edge])
            self._residuals[sid0] += ipflx
            self._residuals[sid1] -= ipflx
