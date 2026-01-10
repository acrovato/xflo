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

    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        # Compute mean value of states
        s = 0.5 * (s0 + s1)

        # Compute primitives, Euler flux and wavespeed
        rho, q, p = self._flu.eval_primitive(s)
        f = self._flu.compute_flux(rho, q, p)
        a = abs(q.dot(n)) + self._flu.eval_speed_sound(rho, p)

        # Compute LF flux
        return (f.dot(n) - 0.5 * a * (s0 - s1)) * l
