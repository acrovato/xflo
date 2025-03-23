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

import vtk

class VtkWriter():
    """Write mesh and results to disk using VTK

    Attributes:
    msh : Mesh object
        Mesh data structure
    """
    def __init__(self, mesh):
        self._msh = mesh

    def write(self, nit=None, results=None):
        """Write to disk
        """
        # Create grid data structure
        grid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        ctypes = []

        # Add points
        vertex_point_map = {}
        vertices = self._msh.get_vertices()
        for i_vtx in range(vertices.shape[0]):
            coords = vertices[i_vtx]
            vertex_point_map[i_vtx] = points.InsertNextPoint(coords[0], coords[1], 0.)
        grid.SetPoints(points)

        # Add field cells
        fcell_cell_map = {}
        fcells = self._msh.get_cells()
        for i_cell in range(fcells.shape[0]):
            # create cell and set number of points
            vertices = fcells[i_cell]
            cell = vtk.vtkPolygon()
            cell.GetPointIds().SetNumberOfIds(vertices.shape[0])
            # add each vertex point
            for i_vtx, vtx in enumerate(vertices):
                cell.GetPointIds().SetId(i_vtx, vertex_point_map[vtx])
            # add cell to grid
            fcell_cell_map[i_cell] = cells.InsertNextCell(cell)
            ctypes.append(cell.GetCellType())
        grid.SetCells(ctypes, cells)

        # Add values at cells
        for name, vals in results.items():
            # fill the values
            values = vtk.vtkDoubleArray()
            for i_cell in range(fcells.shape[0]):
                values.InsertValue(fcell_cell_map[i_cell], vals[i_cell])
            # add values to the grid
            values.SetName(name)
            grid.GetCellData().AddArray(values)

        # Write to file
        nit = '' if nit is None else '_{0:04d}'.format(nit)
        fname = f'{self._msh.get_name()}{nit}.vtu'
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(grid)
        print(f'Writing file: {fname}...', end=' ')
        writer.Write()
        print('done.')
