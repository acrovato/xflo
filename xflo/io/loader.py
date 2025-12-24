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

from xflo.utils.error import XFloError, XFloRuntimeError, XFloFileNotFound
from xflo.structure.mesh import Mesh
import gmsh
import numpy as np

# Gmsh element type to nodes and edges count
ELEMTYPE_NNODES = {2: 3, 3: 4} # Tri, Quad

class GmshLoader():
    """Create mesh and hold data structure
    TODO allow custom sizes and names

    Attributes:
    _name : str
        Name of the mesh
    _has_logged : bool
        Whether log has been written
    """
    def __init__(self, name):
        # Initialize Gmsh and start logger
        self._name = name
        self._has_logged = False

    def __del__(self):
        # Get log and stop Gmsh
        if not self._has_logged:
            self._finalize()

    def create_mesh(self, fname=None):
        """Create the mesh from a set of coordinates

        Parameters:
        fname : str (default : None)
            Path to file containing the coordinates of the airfoil

        Returns:
        msh : Mesh object
            Mesh data structure
        """
        # Infer coordinates file name if not given and load coordinates
        fname = fname if fname else self._name + '.dat'
        coords, le_idx, is_sharp = self._load_coordinates(fname)

        # Initialize Gmsh
        self._initialize()

        # Create geometry and mesh using Gmsh, then load into internal data structure
        self._create_geometry(coords, le_idx, is_sharp)
        self._create_mesh()
        msh = self._build_mesh_data()

        # Finalize Gmsh
        self._finalize()

        return msh

    def _build_mesh_data(self):
        """Build internal mesh data structure

        Returns:
        msh : Mesh object
            Mesh data structure
        """
        # Get coordinates
        vertices = []
        nodes_map = {} # Gmsh node tag to own vertex ID
        nodes_tags, coords, _ = gmsh.model.mesh.get_nodes()
        for i_node, node in enumerate(nodes_tags):
            vertices.append([coords[i_node * 3], coords[i_node * 3 + 1]])
            nodes_map[node] = i_node

        # Get field cells and all edges
        edges_cells = {}
        edges_verts = {}
        cells = []
        i_cell = 0
        elements_map = {} # Gmsh element tag to own field cell ID
        field_grougs = gmsh.model.get_physical_groups(2)
        # Physical group
        for dim, ptag in field_grougs:
            name = gmsh.model.get_physical_name(dim, ptag)
            entites_tag = gmsh.model.get_entities_for_physical_group(dim, ptag)
            # Geometric entity
            for entity in entites_tag:
                el_types, el_tags, el_nodes = gmsh.model.mesh.get_elements(dim, entity)
                # Element type
                for i_type, type in enumerate(el_types):
                    # Number of nodes and edges are the same for 2D cells
                    n_nodes = ELEMTYPE_NNODES[type]
                    n_edges = n_nodes
                    # Build cell connectivity
                    for i_elem, elem in enumerate(el_tags[i_type]):
                        elements_map[elem] = i_cell
                        cell_vtx = []
                        for node in el_nodes[i_type][i_elem * n_nodes:(i_elem + 1) * n_nodes]:
                            cell_vtx.append(nodes_map[node])
                        cells.append(cell_vtx)
                        i_cell += 1
                    # Build edge connectivity
                    edge_nodes_tag = gmsh.model.mesh.get_element_edge_nodes(type, entity)
                    edge_tags, _ = gmsh.model.mesh.get_edges(edge_nodes_tag)
                    for i_edge, edge in enumerate(edge_tags):
                        if not edge in edges_cells:
                            edges_cells[edge] = [elements_map[el_tags[i_type][i_edge // n_edges]]]
                            edges_verts[edge] = [nodes_map[edge_nodes_tag[i_edge * 2]], nodes_map[edge_nodes_tag[i_edge * 2 + 1]]]
                        else:
                            edges_cells[edge].append(elements_map[el_tags[i_type][i_edge // n_edges]])

        # Change edges numbering
        edges_map = {} # Gmsh edge tag to own edge ID
        for i_edge, edge in enumerate(edges_cells):
            edges_map[edge] = i_edge

        # Get internal edges
        field_edges = []
        for edge, neighboor_cells in edges_cells.items():
            if len(neighboor_cells) == 2:
                field_edges.append(edges_map[edge])

        # Get boundary edges
        boundary_edges = {}
        boundary_grougs = gmsh.model.get_physical_groups(1)
        for dim, ptag in boundary_grougs:
            name = gmsh.model.get_physical_name(dim, ptag)
            boundary_edges[name] = []
            etags = gmsh.model.get_entities_for_physical_group(dim, ptag)
            for etag in etags:
                el_types, _, _ = gmsh.model.mesh.get_elements(dim, etag)
                for type in el_types:
                    edge_tags, _ = gmsh.model.mesh.get_edges(gmsh.model.mesh.get_element_edge_nodes(type, etag))
                    for edge in edge_tags:
                        boundary_edges[name].append(edges_map[edge])

        # Create internal data structure
        msh = Mesh(self._name, vertices, cells, list(edges_cells.values()), list(edges_verts.values()), field_edges, boundary_edges)
        return msh

    def _load_coordinates(self, fname):
        """Load and check airfoil coordinates

        Parameters:
        fname : str
            Path to file containing the coordinates of the airfoil

        Return:
        coords : numpy.array
            Airfoil coordinates
        le_idx : int
            Index of leading edge point
        is_sharp : bool
            Whether the airfoil has a sharp or a blunt TE
        """
        # Load coordinates
        try:
            coords = np.loadtxt(fname, skiprows=1)
        except:
            raise XFloFileNotFound(f'File "{fname}" not found!')

        # Check if exactly two columns have been provided
        if coords.shape[1] != 2:
            raise XFloRuntimeError(f'Expected a list of coordinates with 2 columns, but got {coords.shape[1]} instead!')

        # Check if coordinates are in Selig format
        if coords[0, 0] != 1.0 or coords[-1, 0] != 1.0:
            raise XFloRuntimeError('Airfoil coordinates must be ordered using Selig format: TE point must be first and last (duplicated) and its x-coordinate must be equal to 1.0!')

        # Reverse order of coordinates if in standard Selig format to have inward normals
        if coords[1, 1] > coords[-2, 1]:
            coords = np.flipud(coords)

        # Check if airfoil has a sharp or blunt trailing edge
        is_sharp = False
        if coords[0][1] == coords[-1][1]:
            is_sharp = True
            coords = np.delete(coords, (-1), axis=0) # delete duplicated last point

        # Get leading edge index
        le_idx = np.argmin(coords[:, 0])

        return coords, le_idx, is_sharp

    def _create_geometry(self, coords, le_idx, is_sharp):
        """Create geometry in Gmsh

        Parameters:
        coords : numpy.array
            Airfoil coordinates
        le_idx : int
            Index of leading edge point
        is_sharp : bool
            Whether the airfoil has a sharp or a blunt TE
        """
        # Add airfoil points
        airf_ptags = []
        for c in coords:
            airf_ptags.append(gmsh.model.geo.add_point(c[0], c[1], 0.0))
        # Add airfoil spline
        airf_ctags = []
        airf_ctags.append(gmsh.model.geo.add_spline(airf_ptags[0:le_idx + 1]))
        if is_sharp:
            airf_ctags.append(gmsh.model.geo.add_spline(airf_ptags[le_idx:] + [airf_ptags[0]]))
        else:
            airf_ctags.append(gmsh.model.geo.add_spline(airf_ptags[le_idx:]))
            airf_ctags.append(gmsh.model.geo.add_line(airf_ptags[-1], airf_ptags[0]))

        # Add farfield boundary points
        ffc_ptag = gmsh.model.geo.add_point(0., 0., 0.) # center of the sphere
        ff_ptags = []
        ff_ptags.append(gmsh.model.geo.add_point(50., 0., 0.))
        ff_ptags.append(gmsh.model.geo.add_point(0., 50., 0.))
        ff_ptags.append(gmsh.model.geo.add_point(-50., 0., 0.))
        ff_ptags.append(gmsh.model.geo.add_point(0., -50., 0.))
        # Add farfield boundary curves
        ff_ctags = []
        for i in range(4):
            ff_ctags.append(gmsh.model.geo.add_circle_arc(ff_ptags[i], ffc_ptag, ff_ptags[(i+1) % 4]))

        # Add domain field
        fld_tag = gmsh.model.geo.add_plane_surface([gmsh.model.geo.add_curve_loop(ff_ctags),
                                                    gmsh.model.geo.add_curve_loop(airf_ctags)])

        # Add physical group
        gmsh.model.geo.synchronize()
        gmsh.model.add_physical_group(1, airf_ctags, name='airfoil')
        gmsh.model.add_physical_group(1, ff_ctags, name='farfield')
        gmsh.model.add_physical_group(2, [fld_tag], name='field')

        # Add meshing constraints
        gmsh.model.geo.mesh.set_transfinite_curve(airf_ctags[0], 51, 'Bump', coef=0.2)
        gmsh.model.geo.mesh.set_transfinite_curve(airf_ctags[1], 51, 'Bump', coef=0.2)
        if not is_sharp:
            gmsh.model.geo.mesh.set_transfinite_curve(airf_ctags[2], 2)
        for tag in ff_ctags:
            gmsh.model.geo.mesh.set_transfinite_curve(tag, 11)
        #gmsh.model.geo.mesh.set_recombine(2, fld_tag)
        gmsh.model.geo.synchronize()

        # Add boundary layer
        bl_f = gmsh.model.mesh.field.add('BoundaryLayer')
        gmsh.model.mesh.field.set_numbers(bl_f, 'CurvesList', airf_ctags)
        gmsh.model.mesh.field.set_number(bl_f, 'Size', 1e-2)
        gmsh.model.mesh.field.set_number(bl_f, 'Ratio', 1.05)
        gmsh.model.mesh.field.set_number(bl_f, 'Quads', 1)
        gmsh.model.mesh.field.set_number(bl_f, 'Thickness', 1e-1)
        if is_sharp:
            gmsh.model.mesh.field.set_numbers(bl_f, 'FanPointsList', [airf_ptags[0]])
            gmsh.model.mesh.field.set_numbers(bl_f, 'FanPointsSizesList', [10])
        else:
            gmsh.model.mesh.field.set_numbers(bl_f, 'FanPointsList', [airf_ptags[0], airf_ptags[-1]])
            gmsh.model.mesh.field.set_numbers(bl_f, 'FanPointsSizesList', [5])
        gmsh.model.mesh.field.set_as_boundary_layer(bl_f)
        gmsh.model.geo.synchronize()

    def _create_mesh(self):
        """Generate mesh
        """
        import os
        gmsh.option.set_number('Mesh.Algorithm', 6)
        gmsh.option.set_number('Mesh.Optimize', 1)
        gmsh.option.set_number('General.NumThreads', os.cpu_count())
        try:
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.create_edges()
        except Exception as e:
            gmsh.write(self._name + '.msh')
            self._finalize()
            raise XFloError(e)

    def _initialize(self):
        """Start Gmsh and logger
        """
        gmsh.initialize()
        gmsh.logger.start()
        gmsh.model.add(self._name)

    def _finalize(self):
        """Write Gmsh log messages and terminate
        """
        # Get messages and stop logger
        log_msgs = gmsh.logger.get()
        gmsh.logger.stop()
        # Write to file
        file = open('log_gmsh.txt', 'w')
        for m in log_msgs:
            file.write(m + '\n')
        file.close()
        gmsh.finalize()
        self._has_logged = True
