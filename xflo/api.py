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
from xflo.constitutive.perfect_gas import PerfectGas
from xflo.structure.problem import Problem
from xflo.numerics.fv_discretization import FiniteVolume
from xflo.utils import isa
from xflo.utils.error import XFloRuntimeError

def init_xflo(cfg):
    """Initialize xFlo
    TODO add doc

    Parameters:
    cfg : dict
        Dictionary to configure xFlo

    Returns:
    Dictionary of xFlo objects : dict
    """
    # Create and load mesh
    msh_ldr = GmshLoader(cfg['Model']['Name'])
    msh = msh_ldr.create_mesh(cfg['Model']['AirfoilFile'])

    # Create writer
    wrt = VtkWriter(msh)

    # Create perfect gas conservation laws
    fluid = PerfectGas()

    # Create problem and BCs
    pbl = Problem(msh, fluid)
    pbl.set_farfield('farfield')
    bdy = pbl.set_wall('airfoil')

    # Set freestream
    aoa = cfg['Freestream']['AoA']
    mach = cfg['Freestream']['Mach']
    rho, p = isa.eval(cfg['Freestream']['Altitude'])
    pbl.set_freestream(aoa, mach, rho, p)

    # Set geometry
    c_ref = cfg['Geometry']['RefChord']
    x_ref = cfg['Geometry']['RefX']
    z_ref = cfg['Geometry']['RefY']
    pbl.set_geometry(c_ref, x_ref, z_ref)

    # Create spatial discretization scheme
    flux = _select_flux(cfg, fluid)
    grad = _select_grad(cfg, pbl)
    disc = FiniteVolume(pbl, flux, grad)

    # Create time integration method
    tsol = _select_time_integration(cfg, disc, wrt)

    # Return objects
    return {
        'Problem': pbl,
        'Body': bdy,
        'Solver': tsol
    }

def _select_flux(cfg, fluid):
    """Select numerical scheme to calculate the convective flux"""
    method = cfg['Scheme']['Name']
    if method == 'LaxFriedrichs':
        from xflo.numerics.lax_friedrichs import LaxFriedrichs
        return LaxFriedrichs(fluid)
    elif method == 'JST':
        from xflo.numerics.jst import JST
        k2 = cfg['Scheme']['Parameters'].get('k2', 1.)
        k4 = cfg['Scheme']['Parameters'].get('k4', 1./32.)
        c4 = cfg['Scheme']['Parameters'].get('c4', 2.)
        return JST(fluid, k2, k4, c4)
    else:
        raise XFloRuntimeError('Convective scheme not available. Choose between: LaxFriedrichs, JST.')

def _select_grad(cfg, pbl):
    """Select numerical method to reconstruct the gradients"""
    method = cfg.get('Gradient')
    if method is None:
        return None
    elif method == 'GreenGauss':
        from xflo.numerics.green_gauss import GreenGauss
        return GreenGauss(pbl)
    else:
        raise XFloRuntimeError('Gradient reconstruction method not available. Choose between: None, GreenGauss.')

def _select_time_integration(cfg, disc, wrt):
    """Select time integration method"""
    method = cfg['TimeIntegration']['Name']
    if method == 'ExplicitEuler':
        icfl = cfg['TimeIntegration']['Parameters'].get('CflInitial', 1.0)
        rtol = cfg['TimeIntegration']['Parameters'].get('RelativeTolerance', 1e-6)
        mxit = cfg['TimeIntegration']['Parameters'].get('MaxNoIterations', 1000)
        sfrq = cfg['TimeIntegration']['Parameters'].get('SaveFrequency', 100)
        from xflo.numerics.explicit_runge_kutta import ExplicitEuler
        return ExplicitEuler(disc, wrt, icfl, rtol, mxit, sfrq)
    elif method == 'ExplicitRungeKutta4':
        icfl = cfg['TimeIntegration']['Parameters'].get('CflInitial', 1.0)
        rtol = cfg['TimeIntegration']['Parameters'].get('RelativeTolerance', 1e-6)
        mxit = cfg['TimeIntegration']['Parameters'].get('MaxNoIterations', 1000)
        sfrq = cfg['TimeIntegration']['Parameters'].get('SaveFrequency', 100)
        from xflo.numerics.explicit_runge_kutta import ExplicitRk4
        return ExplicitRk4(disc, wrt, icfl, rtol, mxit, sfrq)
    elif method == 'ImplicitEuler':
        icfl = cfg['TimeIntegration']['Parameters'].get('CflInitial', 1.0)
        ecfl = cfg['TimeIntegration']['Parameters'].get('CflExponent', 0.7)
        rtol = cfg['TimeIntegration']['Parameters'].get('RelativeTolerance', 1e-6)
        mxit = cfg['TimeIntegration']['Parameters'].get('MaxNoIterations', 100)
        sfrq = cfg['TimeIntegration']['Parameters'].get('SaveFrequency', 10)
        from xflo.numerics.implicit_euler import ImplicitEuler
        sol = ImplicitEuler(disc, wrt, icfl, ecfl, rtol, mxit, sfrq)
        rtol = cfg['TimeIntegration']['InnerSolver'].get('RelativeTolerance', 1e-3)
        atol = cfg['TimeIntegration']['InnerSolver'].get('AbsoluteTolerance', 1e-5)
        mxit = cfg['TimeIntegration']['InnerSolver'].get('MaxNoIterations', 10)
        nrst = cfg['TimeIntegration']['InnerSolver'].get('NoRestart', 20)
        dtol = cfg['TimeIntegration']['InnerSolver'].get('DropTolerance', 1e-6)
        ffct = cfg['TimeIntegration']['InnerSolver'].get('FillFactor', 20.)
        sol.set_solver_parameters(rtol, atol, mxit, nrst)
        sol.set_preconditioner_parameters(dtol, ffct)
        return sol
    else:
        raise XFloRuntimeError('Time integration method not available. Choose between: ExplicitEuler, ExplicitRungeKutta4, ImplicitEuler.')
