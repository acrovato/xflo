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

from .flux import Flux

class LaxFriedrichs(Flux):
    """Lax-Friedrichs flux formulation
    """
    def __init__(self, fluid):
        super().__init__(fluid)

    def compute_residual(self, state0, state1, n, l):
        # Compute primitives from states
        rho0, q0, p0 = self._flu.eval_primitive(state0)
        rho1, q1, p1 = self._flu.eval_primitive(state1)

        # Compute Euler fluxes
        f0 = self._flu.compute_flux(rho0, q0, p0, state0[3])
        f1 = self._flu.compute_flux(rho1, q1, p1, state1[3])

        # Compute maximum wavespeed
        c0 = self._flu.eval_speed_sound(rho0, p0)
        c1 = self._flu.eval_speed_sound(rho1, p1)
        a = max([q0.dot(n) + c0, q1.dot(n) + c1])

        # Compute LF flux
        return (0.5 * (f0 + f1).dot(n) - 0.5 * a * (state0 - state1)) * l
