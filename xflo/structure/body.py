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

import numpy as np

class Body:
    """Body which will be monitored
    TODO interpolate mach, cp and coords at nodes

    Attributes:
    _name : str
        Name of the body
    _cids : np.array(int)
        IDs of cell connected to this body
    _vcrd : np.array(float)
        Vertices coordinates
    _ecgs : np.array(float)
        Edges center of gravity
    _elgt : np.array(float)
        Edges length
    _enrm : np.array(float)
        Edges unit normla vector
    _mach : np.array(float)
        Mach number
    _cp : np.array(float)
        Pressure coefficient
    _cl : float
        Lift coefficient
    _cd : float
        Drag coefficient
    _cm : float
        Pitching moment coefficient
    """
    def __init__(self, name, mesh):
        # Set name
        self._name = name

        # Init data structure
        bnd = mesh.get_boundary(name)
        coords = mesh.get_vertices()
        cids = mesh.get_edges_cells()
        vids = mesh.get_edges_vertices()

        # Get cells ID
        eids = bnd.get_edge_ids()
        self._cids = np.array([cids[eid][0] for eid in eids])

        # Get edges length and unit normal vector
        self._elgt, self._enrm = bnd.get_edge_metrics()

        # Compute edge center of gravity
        n_edges = self._cids.shape[0]
        self._ecgs = np.zeros((n_edges, 2), dtype=float)
        for i_edge, eid in enumerate(eids):
            self._ecgs[i_edge, :] = 0.5 * (coords[vids[eid, 0]] + coords[vids[eid, 1]])

        # Get vertices coordinates
        self._vcrd = np.zeros((n_edges + 1, 2), dtype=float)
        for i_edge in range(n_edges + 1):
            self._vcrd[i_edge, :] = coords[vids[eids[i_edge % n_edges], 0]]

        # Init variables
        self._mach = np.zeros(n_edges, dtype=float)
        self._cp = np.zeros(n_edges, dtype=float)
        self._cl = 0.
        self._cd = 0.
        self._cm = 0.

    def get_coords(self, at_vertex=False):
        """Get body coordinates

        Parameters:
        at_vertex : bool (default: False)
            Whether to return edges center or vertices coordinates

        Returns:
        coordinates of edge CG : np.array(float)
        """
        if at_vertex:
            return self._vcrd
        else:
            return self._ecgs

    def get_mach(self, at_vertex=False):
        """Get Mach number on body surface

        Parameters:
        at_vertex : bool (default: False)
            Whether to return data at cell centers or interpolated at vertices

        Returns:
        mach number : np.array(float)
        """
        if at_vertex:
            return self._interp(self._mach)
        else:
            return self._mach

    def get_pressure_coef(self, at_vertex=False):
        """Get pressure coefficient on body surface

        Parameters:
        at_vertex : bool (default: False)
            Whether to return data at cell centers or interpolated at vertices

        Returns:
        pressure coefficient : np.array(float)
        """
        if at_vertex:
            return self._interp(self._cp)
        else:
            return self._cp

    def get_lift_coef(self):
        """Returns:
        lift coefficient : float
        """
        return self._cl

    def get_drag_coef(self):
        """Returns:
        drag coefficient : float
        """
        return self._cd

    def get_pitch_coef(self):
        """Returns:
        pitching moment coefficient : float
        """
        return self._cm

    def update(self, cref, xref, aoa, mach, cp):
        """Update variables and aerodynamic load coefficients

        Parameters:
        cref : float
            Reference chord length
        xref : np.array(float)
            Reference center for moment computation
        aoa : float
            Angle of attack
        mach : np.array(float)
            Mach number
        cp : np.array(float)
            Pressure coefficient
        """
        # Get Mach and pressure coefficient
        for i_cell, cid in enumerate(self._cids):
            self._mach[i_cell] = mach[cid]
            self._cp[i_cell] = cp[cid]

        # Compute integrated aerodynamic load coefficients
        cxy = 0.
        cm = 0.
        for i_edge in range(self._cids.shape[0]):
            cxy -= self._cp[i_edge] * self._elgt[i_edge] * self._enrm[i_edge, :]
            cm -= self._cp[i_edge] * self._elgt[i_edge] * (-(self._ecgs[i_edge, 0] - xref[0]) * self._enrm[i_edge, 1] + (self._ecgs[i_edge, 1] - xref[1]) * self._enrm[i_edge, 0])
        self._cl = (cxy[1] * np.cos(aoa) - cxy[0] * np.sin(aoa)) / cref
        self._cd = (cxy[1] * np.sin(aoa) + cxy[0] * np.cos(aoa)) / cref
        self._cm = cm / cref

    def _interp(self, data):
        """Interpolate data at nodes"""
        n_vtx = self._vcrd.shape[0]
        idata = np.zeros(n_vtx)
        for i in range(n_vtx):
            idata[i] = 0.5 * (data[i - 1] + data[i % (n_vtx - 1)])
        return idata
