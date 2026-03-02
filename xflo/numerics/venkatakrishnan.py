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

from .limiter import Limiter
import numpy as np

class Venkatakrishnan(Limiter):
    """Venkatakrishnan limiter
    On the accuracy of limiters and convergence to steady state solutions, V. Venkatakrishnan,
    31st Aerospace Sciences Meeting, AIAA, January 1993,
    https://arc.aiaa.org/doi/10.2514/6.1993-880

    Attributes:
    _k : float
        Limiter coefficient
    """
    def __init__(self, problem, k=5.0):
        self._k = k
        self._rho_min = np.zeros(problem.mesh.get_ncells(), dtype=float)
        self._rho_max = np.zeros(problem.mesh.get_ncells(), dtype=float)
        super().__init__(problem)

    def compute(self, states, gradients):
        # Get connectivity and edge info
        ecids = self._pbl.mesh.get_edges_cells()
        fld = self._pbl.mesh.get_field()
        eids = fld.get_edge_ids()
        _, nrms = fld.get_edge_metrics()
        dsts = fld.get_cell_distances()

        # Compute min/max density of cells and neighboors
        self._rho_min[:] = states[::4]
        self._rho_max[:] = states[::4]
        for eid in eids:
            # get cells and density indices
            cid0 = ecids[eid][0]
            cid1 = ecids[eid][1]
            sid0 = self._pbl.get_uids(cid0)[0]
            sid1 = self._pbl.get_uids(cid1)[0]
            # compute min/max density
            self._rho_min[cid0] = np.minimum(self._rho_min[cid0], states[sid1])
            self._rho_min[cid1] = np.minimum(self._rho_min[cid1], states[sid0])
            self._rho_max[cid0] = np.maximum(self._rho_max[cid0], states[sid1])
            self._rho_max[cid1] = np.maximum(self._rho_max[cid1], states[sid0])

        # Compute density jumps and cells limiter
        sign = [-1, 1]
        for i_edge, eid in enumerate(eids):
            for i_cell, cid in enumerate(ecids[eid]):
                # get states indices
                sid = self._pbl.get_uids(cid)
                # compute density jumps
                dm = sign[i_cell] * gradients[sid[0]].dot(nrms[i_edge]) * 0.5 * dsts[i_edge]
                dp = 1.0
                if dm > 0.:
                    dp = (self._rho_max[cid] - states[sid[0]]) / dm
                elif dm < 0.:
                    dp = (self._rho_min[cid] - states[sid[0]]) / dm
                # compute limiter
                self._lim[sid] = np.minimum(self._lim[sid[0]], self._eval_limiter(dp, dm, dsts[i_edge]))

        return self._lim

    def _eval_limiter(self, dp, dm, d):
        """Evaluate limiter function

        Parameters:
        dp: float
            Jump in downstream density (numerator)
        dm: float
            Jump in upstream density (denominator)
        d: float
            Measure of cell size
        """
        eps2 = (self._k * d) ** 3
        return ((dp ** 2 + eps2) + 2 * dm * dp) / (dp ** 2 + 2 * dm ** 2 + dp * dm + eps2)
