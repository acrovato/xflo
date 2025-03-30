#!/usr/bin/python3
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

from xflo.io.loader import GmshLoader
from xflo.io.writer import VtkWriter

def get_model(name):
    import os.path
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', name)

def main():
    for te in ['sharp', 'blunt']:
        fname = get_model(f'naca0012_{te}.dat')
        # Load mesh
        msh_ldr = GmshLoader(f'naca0012_{te}')
        msh = msh_ldr.create_mesh(fname)
        # Write mesh and cells area
        wrt = VtkWriter(msh)
        res = {'area': msh.get_cells_area()}
        wrt.write(0, res)

if __name__ == '__main__':
    main()
