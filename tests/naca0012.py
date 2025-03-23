#!/usr/bin/python3
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

from xflo.io.loader import GmshLoader
from xflo.io.writer import VtkWriter
from xflo.constitutive.perfect_gas import PerfectGas
from xflo.structure.problem import Problem
from xflo.numerics.lax_friedrichs import LaxFriedrichs
from xflo.numerics.fv_discretization import FiniteVolume
from xflo.numerics.explicit_euler import ExplicitEuler

def get_model(name):
    import os.path
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', name)

def main():
    # Freestream
    aoa = 2.
    mach = 0.1
    rho = 1.225
    p = 101325
    # Geometry
    fname = get_model('naca0012_sharp.dat')
    c_ref = 1
    x_ref = 0.25
    z_ref = 0.
    # Numerics
    init_cfl = 0.25
    rel_tol = 1e-3
    max_iter = 10000

    # Load mesh
    msh_ldr = GmshLoader('naca0012')
    msh = msh_ldr.create_mesh(fname)
    # Create writer
    wrt = VtkWriter(msh)

    # Create perfect gas fluid
    fluid = PerfectGas()

    # Create problem
    pbl = Problem(msh, fluid)
    pbl.set_farfield('farfield')
    pbl.set_wall('airfoil')
    pbl.set_freestream(aoa, mach, rho, p)
    pbl.set_geometry(c_ref, x_ref, z_ref)

    # Create spatial discretization scheme
    flux = LaxFriedrichs(fluid)
    disc = FiniteVolume(pbl, flux)

    # Create time integration method
    tint = ExplicitEuler(disc, wrt, init_cfl, rel_tol, max_iter)

    # Solve flow
    tint.run()

if __name__ == '__main__':
    main()
