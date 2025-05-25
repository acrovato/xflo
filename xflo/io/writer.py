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

from xflo.utils.log import logger
import vtk

class VtkWriter():
    """Write mesh and results to disk using VTK

    Attributes:
    _name : str
        Base name of case to save
    _grid : vtk.vtkUnstructuredGrid
        VTK grid
    _writer : vtk.vtkXMLUnstructuredGridWriter
        VTK grid writer
    _cell_cell_map : dict(int, int)
        Map between xFlo and VTK cells ID
    """
    def __init__(self, mesh):
        """Parameters:
        mesh : xflo.structure.mesh.Mesh
            xFlo msh data structure
        """
        self._name = mesh.get_name()

        # Create grid data structure
        self._grid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        ctypes = []

        # Add points
        vertex_point_map = {}
        vertices = mesh.get_vertices()
        for i_vtx in range(vertices.shape[0]):
            coords = vertices[i_vtx]
            vertex_point_map[i_vtx] = points.InsertNextPoint(coords[0], coords[1], 0.)
        self._grid.SetPoints(points)

        # Add field cells
        self._cell_cell_map = {}
        fcells = mesh.get_cells()
        self._ncells = fcells.shape[0]
        for i_cell in range(self._ncells):
            # create cell and set number of points
            vertices = fcells[i_cell]
            cell = vtk.vtkPolygon()
            cell.GetPointIds().SetNumberOfIds(vertices.shape[0])
            # add each vertex point
            for i_vtx, vtx in enumerate(vertices):
                cell.GetPointIds().SetId(i_vtx, vertex_point_map[vtx])
            # add cell to grid
            self._cell_cell_map[i_cell] = cells.InsertNextCell(cell)
            ctypes.append(cell.GetCellType())
        self._grid.SetCells(ctypes, cells)

        # Create writer
        self._writer = vtk.vtkXMLUnstructuredGridWriter()
        self._writer.SetInputData(self._grid)

    def write(self, nit=None, results=None):
        """Write mesh and solution to disk

        Parameters:
        nit : int (default: None)
            Iteration at which the file is writter
        results : dict(str: array(float)) (default: None)
            Dictionay containing field names and values to write
        """
        # Add values at cells
        for name, vals in results.items():
            # fill the values
            values = vtk.vtkDoubleArray()
            for i_cell in range(self._ncells):
                values.InsertValue(self._cell_cell_map[i_cell], vals[i_cell])
            # add values to the grid
            values.SetName(name)
            self._grid.GetCellData().AddArray(values)

        # Write to file
        nit = '' if nit is None else '_{0:04d}'.format(nit)
        fname = f'{self._name}{nit}.vtu'
        self._writer.SetFileName(fname)
        logger.info(f'Writing file: {fname}')
        self._writer.Write()
