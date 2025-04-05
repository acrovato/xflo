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
        # Compute mean value of states
        state = 0.5 * (state0 + state1)

        # Compute primitives, Euler flux and wavespeed
        rho, q, p = self._flu.eval_primitive(state)
        f = self._flu.compute_flux(rho, q, p, state[3])
        a = q.dot(n) + self._flu.eval_speed_sound(rho, p)

        # Compute LF flux
        return (f.dot(n) - 0.5 * a * (state0 - state1)) * l
