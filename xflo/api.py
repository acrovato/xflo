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

    Parameters:
    cfg : dict
        Dictionary to configure xFlo

    Returns:
    Dictionary of xFlo objects : dict

    Examples:
        >>> cfg = {
        >>>     'Model': {
        >>>         'Name': str, # model name
        >>>         'AirfoilFile': str, # path to file containing airfoil coordinates
        >>>         'Parameters': {
        >>>             'NumCells': int, # number of cells on the pressure and suction sides
        >>>             'Bump': float, # factor controlling the refinement at the leading and trailing edges
        >>>         }
        >>>     },
        >>>     'Freestream': {
        >>>         'AoA': float, # angle of attack (degrees)
        >>>         'Mach': float, # Mach number
        >>>         'Altitude': float, # altitude expressed as flight level (e.g. 100 is 10000 ft)
        >>>     },
        >>>     'Geometry': {
        >>>         'RefChord': float, # reference chord length
        >>>         'RefX': flot, # reference x-coordinate
        >>>         'RefY': float, # reference y-coordinate
        >>>     },
        >>>     'Scheme': {
        >>>         'Name': str, # convective scheme name ("LaxFriedrichs", "JST", "Roe", "HLL", "HLLC")
        >>>         'Parameters': dict, # parameters to configure the convective scheme, see their respective class
        >>>     },
        >>>     # Only required for second-order accurate schemes
        >>>     'Gradient': {
        >>>         'Name': str, # gradient reconstruction algorithm ("GreenGauss")
        >>>         # Only required for upwind schemes
        >>>         'Limiter': {
        >>>             'Name': str, # limiter method name ("Venkatakrishnan")
        >>>             'Parameters': dict, # parameters to configure the limiter, see their respective class
        >>>         }
        >>>     },
        >>>     'TimeIntegration': {
        >>>         'Name': str, # time integration method ("ExplicitEuler", "ExplicitRungeKutta4", "ImplicitEuler")
        >>>         'Parameters': {
        >>>             'CflInitial': float, # initial CFL number
        >>>             'CflExponent': float, # exponent in CFL adaptation law
        >>>             'RelativeTolerance': float, # relative tolerance on density residual
        >>>             'MaxNumIterations': int, # maximum number of iterations
        >>>             'SaveFrequency': int, # interval at which to save solution to disk
        >>>         },
        >>>         # Only required for implicit methods
        >>>         'InnerSolver': {
        >>>             'RelativeTolerance': float, # relative tolerance for GMRES solver
        >>>             'AbsoluteTolerance': float, # absolute tolerance for GMRES solver
        >>>             'MaxNumIterations': int, # maximum number of iterations for GMRES solver
        >>>             'NumRestart': int, # number of iterations between restart for GMRES solver (Krylov maximum subspace size)
        >>>             'DropTolerance': float, # drop tolerance for ILU preconditioner
        >>>             'FillFactor': float, # fill ratio upper bound for ILU preconditioner
        >>>         }
        >>>     }
        >>> }
    """
    # Create and load mesh
    msh_ldr = GmshLoader(cfg['Model']['Name'])
    n_cells, bump = _get_mesh_parameters(cfg['Model'].get('Parameters', {}))
    msh = msh_ldr.create_mesh(cfg['Model']['AirfoilFile'], num_cells=n_cells, bump=bump)

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
    flux = _select_flux(cfg['Scheme'], fluid)
    grad = _select_gradient(cfg.get('Gradient', {}), pbl)
    disc = FiniteVolume(pbl, flux, grad)

    # Create time integration method
    tsol = _select_time_integration(cfg['TimeIntegration'], disc, wrt)

    # Return objects
    return {
        'Problem': pbl,
        'Body': bdy,
        'Solver': tsol
    }

def _get_mesh_parameters(cfg):
    """Get (default) mesh parameters"""
    return cfg.get('NumCells', 50), cfg.get('Bump', 0.2)

def _select_flux(cfg, fluid):
    """Select numerical scheme to calculate the convective flux"""
    method = cfg['Name']
    if method == 'LaxFriedrichs':
        from xflo.numerics.lax_friedrichs import LaxFriedrichs
        return LaxFriedrichs(fluid)
    elif method == 'JST':
        from xflo.numerics.jst import JST
        k2 = cfg['Parameters'].get('k2', 1.)
        k4 = cfg['Parameters'].get('k4', 1./32.)
        c4 = cfg['Parameters'].get('c4', 2.)
        return JST(fluid, k2, k4, c4)
    elif method == 'Roe':
        from xflo.numerics.roe import Roe
        efix = cfg['Parameters'].get('EntropyFix', 1e-6)
        return Roe(fluid, efix)
    elif method == 'HLL':
        from xflo.numerics.hll import HLL
        return HLL(fluid)
    elif method == 'HLLC':
        from xflo.numerics.hllc import HLLC
        return HLLC(fluid)
    else:
        raise XFloRuntimeError('Convective scheme not available. Choose between: "LaxFriedrichs", "JST", "Roe", "HLL", "HLLC".')

def _select_limiter(cfg, pbl):
    """Select method to limit the gradients"""
    method = cfg.get('Name')
    if method is None:
        return None
    elif method == 'Venkatakrishnan':
        from xflo.numerics.venkatakrishnan import Venkatakrishnan
        k = cfg.get('Parameters', {}).get('Coefficient', 5.0)
        return Venkatakrishnan(pbl, k)
    else:
        raise XFloRuntimeError('Limiter not available. Choose between: None, "Venkatakrishnan".')

def _select_gradient(cfg, pbl):
    """Select numerical method to reconstruct the gradients"""
    method = cfg.get('Name')
    if method is None:
        return None
    else:
        lim = _select_limiter(cfg.get('Limiter', {}), pbl)
        if method == 'GreenGauss':
            from xflo.numerics.green_gauss import GreenGauss
            return GreenGauss(pbl, lim)
        else:
            raise XFloRuntimeError('Gradient reconstruction method not available. Choose between: None, "GreenGauss".')

def _select_time_integration(cfg, disc, wrt):
    """Select time integration method"""
    method = cfg['Name']
    if method == 'ExplicitEuler':
        icfl = cfg['Parameters'].get('CflInitial', 1.0)
        rtol = cfg['Parameters'].get('RelativeTolerance', 1e-6)
        mxit = cfg['Parameters'].get('MaxNumIterations', 1000)
        sfrq = cfg['Parameters'].get('SaveFrequency', 100)
        from xflo.numerics.explicit_runge_kutta import ExplicitEuler
        return ExplicitEuler(disc, wrt, icfl, rtol, mxit, sfrq)
    elif method == 'ExplicitRungeKutta4':
        icfl = cfg['Parameters'].get('CflInitial', 1.0)
        rtol = cfg['Parameters'].get('RelativeTolerance', 1e-6)
        mxit = cfg['Parameters'].get('MaxNumIterations', 1000)
        sfrq = cfg['Parameters'].get('SaveFrequency', 100)
        from xflo.numerics.explicit_runge_kutta import ExplicitRk4
        return ExplicitRk4(disc, wrt, icfl, rtol, mxit, sfrq)
    elif method == 'ImplicitEuler':
        icfl = cfg['Parameters'].get('CflInitial', 1.0)
        ecfl = cfg['Parameters'].get('CflExponent', 0.7)
        rtol = cfg['Parameters'].get('RelativeTolerance', 1e-6)
        mxit = cfg['Parameters'].get('MaxNumIterations', 100)
        sfrq = cfg['Parameters'].get('SaveFrequency', 10)
        from xflo.numerics.implicit_euler import ImplicitEuler
        sol = ImplicitEuler(disc, wrt, icfl, ecfl, rtol, mxit, sfrq)
        rtol = cfg['InnerSolver'].get('RelativeTolerance', 1e-3)
        atol = cfg['InnerSolver'].get('AbsoluteTolerance', 1e-5)
        mxit = cfg['InnerSolver'].get('MaxNumIterations', 10)
        nrst = cfg['InnerSolver'].get('NumRestart', 20)
        dtol = cfg['InnerSolver'].get('DropTolerance', 1e-6)
        ffct = cfg['InnerSolver'].get('FillFactor', 20.)
        sol.set_solver_parameters(rtol, atol, mxit, nrst)
        sol.set_preconditioner_parameters(dtol, ffct)
        return sol
    else:
        raise XFloRuntimeError('Time integration method not available. Choose between: "ExplicitEuler", "ExplicitRungeKutta4", "ImplicitEuler".')
