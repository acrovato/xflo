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

class Group:
    """Physical group part of a mesh

    Attributes:
    _name : str
        Name of the group
    _eids : np.array(int), size: (n_edges)
        Group edges ID
    _elgt : np.array(float), size: (n_edges)
        Edges length
    _enrm : np.array(float), size: (n_edges, 2)
        Edges unit normal vector
    _edst : np.array(float), size: (n_edges)
        Distance between neighboor cells
    """
    def __init__(self, name, edge_ids, edge_vertices, edge_cells, cell_vertices, vertices, sort=False):
        """Parameters:
        name : str
            Name of the group
        edge_ids : list(int)
            Group edges ID
        edge_vertices : list(list(int))
            Edges to vertices connectivity
        edge_cells : np.array(np.array(int)), size: (n_edges, n_cells_edge)
            Edges to cells connectivity
        cell_vertices : np.array(np.array(int)), size: (n_cells, n_nodes_cell)
            Cells to vertices connectivity
        vertices : np.array(float), size: (n_nodes, 2)
            Coordinates of the mesh vertices
        sort : bool
            Whether the edges of the group should be sorted
        """
        self._name = name
        n_edges = len(edge_ids)

        # Sort edges
        if sort:
            # only retain edges belonging to current group
            cur_edge_vertices = [edge_vertices[eid] for eid in edge_ids]
            # map row to edge_id
            row_map = {vids[0] : i_edge for i_edge, vids in enumerate(cur_edge_vertices)}
            # sort against connectivity
            j_edge = 0
            self._eids = np.zeros(n_edges, dtype=int)
            for i_edge in range(n_edges):
                self._eids[i_edge] = edge_ids[j_edge]
                j_edge = row_map[cur_edge_vertices[j_edge][1]]
        else:
            self._eids = edge_ids

        # Compute owner cell CG
        cg = np.zeros((n_edges, 2), dtype=float)
        for i_edge in range(n_edges):
            for vid in cell_vertices[edge_cells[self._eids[i_edge]][0]]:
                cg[i_edge] += vertices[vid]
            cg[i_edge] /= cell_vertices[edge_cells[self._eids[i_edge]][0]].shape[0]

        # Compute edges length, unit normal vector and cell-cell distance
        self._elgt = np.zeros(n_edges, dtype=float)
        self._enrm = np.zeros((n_edges, 2), dtype=float)
        self._edst = np.zeros(n_edges, dtype=float)
        for i_edge in range(n_edges):
            xy0 = vertices[edge_vertices[self._eids[i_edge]][0]]
            xy1 = vertices[edge_vertices[self._eids[i_edge]][1]]
            self._elgt[i_edge] = np.linalg.norm(xy1 - xy0)
            self._enrm[i_edge, :] = np.array([-(xy1[1] - xy0[1]), xy1[0] - xy0[0]]) / self._elgt[i_edge]
            self._edst[i_edge] = 2 * np.linalg.norm(0.5 * (xy0 + xy1) - cg[i_edge])

    def get_edge_ids(self):
        """Return:
        group edges id : np.array(int)
        """
        return self._eids

    def get_edge_metrics(self):
        """Returns:
        group edges length : np.array(float)
        group edges unit normal vector : np.array(float)
        """
        return self._elgt, self._enrm

    def get_cell_distances(self):
        """Returns:
        group edges inter-cell distance : np.array(float)
        """
        return self._edst
