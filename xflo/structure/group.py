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
    """
    def __init__(self, name, edge_ids, edge_vertices, vertices, sort=False):
        """Parameters:
        name : str
            Name of the group
        edge_ids : list(int)
            Group edges ID
        edge_vertices : list(list(int))
            Edges to vertices connectivity
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

        # Compute edges length and unit normal vector
        self._elgt = np.zeros(n_edges, dtype=float)
        self._enrm = np.zeros((n_edges, 2), dtype=float)
        for i_edge in range(n_edges):
            xy0 = vertices[edge_vertices[self._eids[i_edge]][0]]
            xy1 = vertices[edge_vertices[self._eids[i_edge]][1]]
            self._elgt[i_edge] = np.linalg.norm(xy1 - xy0)
            self._enrm[i_edge, :] = np.array([-(xy1[1] - xy0[1]), xy1[0] - xy0[0]]) / self._elgt[i_edge]

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
