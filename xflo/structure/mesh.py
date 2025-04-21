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

from .group import Group
import numpy as np

class Mesh:
    """Internal mesh data structure

    Attributes:
    _name : str
        Name of the mesh
    _verts : np.array(float), size: (n_nodes, 2)
        Coordinates of the mesh vertices
    _cells : np.array(np.array(int)), size: (n_cells, n_nodes_cell)
        Field cells to vertices connectivity
    _edg_cel : np.array(np.array(int)), size: (n_edges, n_cells_edge)
        Edges to cells connectivity
    _edg_vrt : np.array(int), size: (n_edges, 2)
        Edges to vertices connectivity
    _field : xflo.group.Group
        Field group
    _field : dict(str: xflo.group.Group)
        Boundary groups
    _carea : np.array(float), size: (n_cells)
        Field cells area
    """
    def __init__(self, name, vertices, cells, edges_cells, edge_vertices, field_edges, boundary_edges):
        """Parameters:
        name : str
            Name of the mesh
        vertices : list(list(float)), size: (n_nodes, 2)
            Coordinates of the mesh vertices
        cells : list(list(int)), size: (n_cells, n_nodes_cell)
            Field cells to vertices connectivity
        edges_cells : list(list(int)), size: (n_edges, n_cells_edge)
            Edges to cells connectivity
        edge_vertices : list(list(int)), size: (n_edges, 2)
            Edges to vertices connectivity
        field_edges : list(int), size: (n_edges)
            Field edges ID
        boundary_edges : dict(str: list(int)), size: (n_boundaries, n_edges)
            Boundary edges ID
        """
        # Init data structure
        self._name = name
        self._verts = np.array(vertices, dtype=float)
        self._cells = np.array([np.array(cell, dtype=int) for cell in cells], dtype=np.ndarray)
        self._edg_cel = np.array([np.array(edge, dtype=int) for edge in edges_cells], dtype=np.ndarray)
        self._edg_vrt = np.array(edge_vertices, dtype=int)

        # Create groups
        self._fld = Group('field', field_edges, edge_vertices, self._edg_cel, self._cells, self._verts)
        self._bnd = {}
        for bnd_name, bnd_edgs in boundary_edges.items():
            self._bnd[bnd_name] = Group(bnd_name, bnd_edgs, edge_vertices, self._edg_cel, self._cells, self._verts, sort=True)

        # Compute cells area
        self._carea = np.zeros(self._cells.shape[0])
        for i_cell in range(self._cells.shape[0]):
            area = 0
            for i_vtx in range(self._cells[i_cell].shape[0]):
                xy0 = self._verts[self._cells[i_cell][i_vtx]]
                xy1 = self._verts[self._cells[i_cell][(i_vtx + 1) % self._cells[i_cell].shape[0]]]
                area += xy0[0] * xy1[1] - xy0[1] * xy1[0]
            self._carea[i_cell] = abs(area) / 2

    def get_name(self):
        """Returns:
        mesh name : str
        """
        return self._name

    def get_ncells(self):
        """Returns:
        number of cells : int
        """
        return self._cells.shape[0]

    def get_nedges(self):
        """Returns:
        number of edges : int
        """
        return self._edg_cel.shape[0]

    def get_vertices(self):
        """Returns:
        vertices coordinates : np.array(float)
        """
        return self._verts

    def get_cells(self):
        """Returns:
        cells connectivity : np.array(np.array(int))
        """
        return self._cells

    def get_edges_cells(self):
        """Returns:
        edges to cells connectivity : np.array(np.array(int))
        """
        return self._edg_cel

    def get_edges_vertices(self):
        """Returns:
        edges to vertices connectivity : np.array(np.array(int))
        """
        return self._edg_vrt

    def get_cells_area(self):
        """Returns:
        cells area : np.array(float)
        """
        return self._carea

    def get_field(self):
        """Returns:
        field group : xflo.group.Group
        """
        return self._fld

    def get_boundary(self, name):
        """Parameters:
        groupe name : str

        Returns:
        boundary group : xflo.group.Group
        """
        return self._bnd[name]
