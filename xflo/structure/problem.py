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

from xflo.boundary.riemann_invariant import RiemannInvariant
from xflo.boundary.slip_wall import SlipWall
from xflo.structure.body import Body
from xflo.structure.variables import Variables
import numpy as np

class Problem:
    """Manage aerodynamic problem

    Attributes:
    mesh : xflo.structure.mesh.Mesh
        Mesh data structure
    _flu : xflo.constitutive.perfect_gas.PerfectGas
        Perfect gas constitutive law equations
    _far : xflo.boundary.riemann_invariant.RiemannInvariant
        Farfield boundary condition
    _wal : list(xflo.boundary.slip_wall.SlipWall)
        Wall boundary conditions
    _bdy : list(xflo.structure.body.Body
        Bodies to monitor
    _var : xflo.structure.variables.Variables
        Flow variables and residuals
    _aoa : float
        Angle of attack
    _minf : float
        Freestream mach number
    _rinf : float
        Freestream density
    _pinf : float
        Freestream pressure
    _cref : float
        Reference chord length
    _xref : np.array(float)
        Reference center for moment computation
    _cl : float
        Lift coefficient
    _cd : float
        Drag coefficient
    _cm : float
        Pitching moment coefficient
    """
    def __init__(self, mesh, fluid):
        # Data objects
        self.mesh = mesh
        self._flu = fluid
        self._far = None
        self._wal = []
        self._bdy = []
        self._var = Variables(mesh.get_ncells())

        # Freestream
        self._aoa = 0.
        self._minf = 0.3
        self._rinf = 1.225
        self._pinf = 101325

        # Reference geometry
        self._cref = 1.
        self._xref = np.array([0.25, 0.])

        # Aerodynamic loads
        self._cl = 0.
        self._cd = 0.
        self._cm = 0.

    def get_bcs(self):
        """Returns:
        boundary conditions : list(xflo.boundary.boundary_condition.BoundaryCondition)
        """
        return [self._far, *self._wal]

    def get_variables(self, name=None):
        """Get variables by name (see xflo.structure.variables)

        Parameters:
        name : str (default: None)
            Name of the variable

        Returns:
        variables : dict(str: np.array(float)) or np.array(float)
            Dictionary of named variables (if name is None), or requested variable array"""
        return self._var.get(name)

    def get_freestream_state(self):
        """Compute freestream conservative variables

        Returns:
        freestream states : np.array(float)
        """
        cinf = self._flu.eval_speed_sound(self._rinf, self._pinf)
        uinf = self._minf * cinf * np.cos(self._aoa)
        vinf = self._minf * cinf * np.sin(self._aoa)
        return self._flu.eval_state(np.array([self._rinf, uinf, vinf, self._pinf]))

    def get_lift_coef(self):
        """Returns:
        lift coefficient : float
        """
        return self._cl

    def get_drag_coef(self):
        """Returns:
        drag coefficient : float
        """
        return self._cd

    def get_pitch_coef(self):
        """Returns:
        pitching moment coefficient : float
        """
        return self._cm

    def set_farfield(self, name):
        """Create farfield boundary condition

        Parameters:
        name : str
            Name of the physical group on which the boundary condition is defined
        """
        self._far = RiemannInvariant(self._flu, name)

    def set_wall(self, name):
        """Create wall boundary condition and monitor it

        Parameters:
        name : str
            Name of the physical group on which the boundary condition is defined

        Returns:
        body : xflo.structure.body.Body
            Body associated to wall boundary conditon which will be monitored
        """
        self._wal.append(SlipWall(self._flu, name))
        self._bdy.append(Body(name, self.mesh))
        return self._bdy[-1]

    def set_freestream(self, aoa, mach, rho, p):
        """Set freestream flow variables and update boundary condition

        Parameters:
        aoa : float
            Angle of attack
        mach : float
            Mach number
        rho : float
            Density
        p : float
            Pressure
        """
        # Set freestream state
        self._aoa = aoa * np.pi / 180.
        self._minf = mach
        self._rinf = rho
        self._pinf = p
        self._far.set_freestream(self._aoa, mach, rho, p)

    def set_geometry(self, c_ref, x_ref, z_ref):
        """Set reference geometry variables

        Parameters:
        c_ref : float
            Reference chord length
        x_ref : np.array(float)
            Reference center
        """
        self._cref = c_ref
        self._xref[0] = x_ref
        self._xref[1] = z_ref

    def update(self, states, residuals, get_ids):
        """Update flow loads

        Parameters:
        states : np.array(float)
            Conservative variables vector
        residuals : np.array(float)
            Residuals vector
        get_ids : method
            Method to get unknown IDs corresponding to a cell ID
        """
        # Update variables and residuals
        cinf = self._flu.eval_speed_sound(self._rinf, self._pinf)
        rqinf = 0.5 * self._rinf * self._minf * cinf * self._minf * cinf
        for i_cell in range(self.mesh.get_ncells()):
            ids = get_ids(i_cell)
            # primitives
            rho, q, p = self._flu.eval_primitive(states[ids])
            u = q[0]
            v = q[1]
            # derived
            c = self._flu.eval_speed_sound(rho, p)
            s = self._flu.eval_entropy(rho, c)
            mach = np.linalg.norm(q) / c
            cp = (p - self._pinf) / rqinf
            # residuals
            res = residuals[ids]
            # update
            self._var.set(i_cell, rho, u, v, p, c, s, mach, cp, res[0], res[1], res[2], res[3])

        # Update variables on body and loads
        self._cl = 0.
        self._cd = 0.
        self._cm = 0.
        for bdy in self._bdy:
            bdy.update(self._cref, self._xref, self._aoa, self._var.get('Mach'), self._var.get('PressureCoefficient'))
            self._cl += bdy.get_lift_coef()
            self._cd += bdy.get_drag_coef()
            self._cm += bdy.get_pitch_coef()
