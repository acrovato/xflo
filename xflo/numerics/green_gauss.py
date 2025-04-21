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

class GreenGauss(Gradient):
    """Green-Gauss method to compute the gradients
    """
    def compute(self, states):
        # Get the edge to cell connectivity and the cells area
        ecids = self._pbl.mesh.get_edges_cells()
        carea = self._pbl.mesh.get_cells_area()

        # Interpolate states at edges
        self._states[:] = 0.
        for i_edge, cids in enumerate(ecids):
            for cid in cids:
                esid = self._pbl.get_uids(i_edge)
                csid = self._pbl.get_uids(cid)
                self._states[esid] += states[csid] / cids.shape[0]

        # Reset gradients
        self._grads[:] = 0.

        # Compute gradients on cells linked to boundary
        for bc in self._pbl.get_bcs():
            bnd = self._pbl.mesh.get_boundary(bc.get_name())
            eids = bnd.get_edge_ids()
            lgts, nrms = bnd.get_edge_metrics()
            for i_edge, eid in enumerate(eids):
                esid = self._pbl.get_uids(eid) # states ID for edge
                csid = self._pbl.get_uids(ecids[eid][0]) # states ID for cell
                self._grads[csid, :] -= self._compute_flux(self._states[esid], nrms[i_edge], lgts[i_edge]) / carea[ecids[eid][0]] # inward normal

        # Compute gradients in the field
        fld = self._pbl.mesh.get_field()
        eids = fld.get_edge_ids()
        lgts, nrms = fld.get_edge_metrics()
        for i_edge, eid in enumerate(eids):
            esid = self._pbl.get_uids(eid) # states ID for edge
            csid0 = self._pbl.get_uids(ecids[eid][0]) # states ID for owner cell
            csid1 = self._pbl.get_uids(ecids[eid][1]) # states ID for neighboor cell
            flux = self._compute_flux(self._states[esid], nrms[i_edge], lgts[i_edge])
            self._grads[csid0, :] -= flux / carea[ecids[eid][0]] # cell owning the edge has inward normal
            self._grads[csid1, :] += flux / carea[ecids[eid][1]] # neighboor cell has outward normal

        return self._grads
