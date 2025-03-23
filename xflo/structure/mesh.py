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

# TODO should we store data structure in dedicated classes Vertex, Cell, Edge, etc.

class Mesh:
    """Internal mesh data structure

    Attributes:
    _name : str
        Name of the mesh
    _ncells : int
        Number of field cells
    _verts : np.array(float), size: (n_nodes, 2)
        Coordinates of the mesh vertices
    _cells : np.array(np.array(int)), size: (n_cells, n_nodes_cell)
        Field cells connectivity
    _edges : np.array(np.array(int)), size: (n_edges, n_cells_edge)
        Edges connectivity
    _f_edges : np.array(int), size: (n_edges, 2)
        Index of internal edges
    _b_edges : dict(str: np.array(int))
        Index of boundary edges
    _c_area : np.array(float), size: (n_cells)
        Field cells area
    _e_lgt : np.array(float), size: (n_edges)
        Edges length
    _e_nrm : np.array(float), size: (n_edges)
        Edges unit normal vector
    """
    def __init__(self, name, vertices, cells, edges, edge_nodes, field_edges, boundary_edges):
        # Init
        self._name = name
        self._ncells = len(cells)
        self._verts = np.array(vertices, dtype=float)
        self._cells = np.array([np.array(cell, dtype=int) for cell in cells], dtype=np.ndarray)
        self._edges = np.array([np.array(edge, dtype=int) for edge in edges], dtype=np.ndarray)
        self._f_edges = np.array(field_edges, dtype=int)
        self._b_edges = {}
        for bnd_name, bnd_edgs in boundary_edges.items():
            self._b_edges[bnd_name] = np.array(bnd_edgs, dtype=int)

        # Compute cells area
        self._c_area = np.zeros(self._cells.shape[0])
        for i_cell in range(self._cells.shape[0]):
            area = 0
            for i_vtx in range(self._cells[i_cell].shape[0]):
                xy0 = self._verts[self._cells[i_cell][i_vtx]]
                xy1 = self._verts[self._cells[i_cell][(i_vtx + 1) % self._cells[i_cell].shape[0]]]
                area += xy0[0] * xy1[1] - xy0[1] * xy1[0]
            self._c_area[i_cell] = abs(area) / 2

        # Compute edges length and unit normal vector
        self._e_lgt = np.zeros(self._edges.shape[0])
        self._e_nrm = np.zeros((self._edges.shape[0], 2))
        for i_edge in range(self._edges.shape[0]):
            xy0 = self._verts[edge_nodes[i_edge][0]]
            xy1 = self._verts[edge_nodes[i_edge][1]]
            self._e_lgt[i_edge] = np.linalg.norm(xy1 - xy0)
            self._e_nrm[i_edge, :] = np.array([-(xy1[1] - xy0[1]), xy1[0] - xy0[0]]) / self._e_lgt[i_edge]

    def get_name(self):
        """Returns:
        mesh name : str
        """
        return self._name

    def get_ncells(self):
        """Returns:
        number of cells : int
        """
        return self._ncells

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

    def get_edges(self):
        """Returns:
        edges connectivity : np.array(np.array(int))
        """
        return self._edges

    def get_cells_area(self):
        """Returns:
        cells area : np.array(float)
        """
        return self._c_area

    def get_field_edges(self):
        """Returns:
        field edges connectivity : np.array(np.array(int))
        field edges length : np.array(float)
        field edges unit normal vector : np.array(float)
        """
        return self._edges[self._f_edges], self._e_lgt[self._f_edges], self._e_nrm[self._f_edges, :]

    def get_boundary_edges(self, boundary_name):
        """
        Arguments:
        name : str
            Name of the boundary

        Returns:
        boundary edges connectivity : np.array(np.array(int))
        boundary edges length : np.array(float)
        boundary edges unit normal vector : np.array(float)
        """
        return self._edges[self._b_edges[boundary_name]], self._e_lgt[self._b_edges[boundary_name]], self._e_nrm[self._b_edges[boundary_name], :]
