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

from .boundary_condition import BoundaryCondition
import numpy as np

class RiemannInvariant(BoundaryCondition):
    """Farfield boundary condition
    Use Riemann invariants to update variables automatically
    Reference: https://fun3d.larc.nasa.gov/papers/NASA-TM-2011-217181.pdf

    Attributes:
    _rinf : float
        Freestream density
    _cinf : float
        Freestream speed of sound
    _qinf : np.array(2)
        Freestream velocity vector
    _is_supersonic : bool
        Whether freestream flow is supersonic
    """
    def __init__(self, fluid, boundary_name):
        super().__init__(fluid, boundary_name)

    def set_freestream(self, aoa, mach, rho, p):
        """Set freestream variables

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
        self._rinf = rho
        self._cinf = self._flu.eval_speed_sound(rho, p)
        self._qinf = mach * self._cinf * np.array([np.cos(aoa), np.sin(aoa)])
        self._is_supersonic = mach > 1.

    def compute_ghost(self, state, n):
        # Reverse direction to get outward normal so that it matches the reference textbook
        n = -n

        # Compute freestream normal velocity
        qn_inf = self._qinf.dot(n)
        is_inflow = qn_inf < 0.

        # Compute primitive variables inside domain
        rho_dom, q_dom, p_dom = self._flu.eval_primitive(state)
        qn_dom = q_dom.dot(n)
        c_dom = self._flu.eval_speed_sound(rho_dom, p_dom)

        # Compute Riemann invariants
        r_p = self._flu.eval_riemann(qn_dom, c_dom, 1)
        r_m = self._flu.eval_riemann(qn_inf, self._cinf, -1)
        if self._is_supersonic:
            if is_inflow:
                r_p = self._flu.eval_riemann(qn_inf, self._cinf, 1) # supersonic inflow
            else:
                r_m = self._flu.eval_riemann(qn_dom, c_dom, -1) # supersonic outflow

        # Compute new edge normal velocity and speed of sound
        u, c = self._flu.eval_speed_riemann(r_p, r_m)

        # Compute velocity and entropy for inflow or outflow
        if is_inflow:
            q = self._qinf + (u - qn_inf) * n
            s = self._flu.eval_entropy(self._rinf, self._cinf)
        else:
            q = q_dom + (u - qn_dom) * n
            s = self._flu.eval_entropy(rho_dom, c_dom)

        # Compute state
        rho, rhoe = self._flu.eval_states_entropy(s, c, q)
        return np.array([rho, rho * q[0], rho * q[1], rhoe])
