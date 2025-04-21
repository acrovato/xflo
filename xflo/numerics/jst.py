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

class JST(Flux):
    """Jameson-Schmidt-Turkel flux formulation
    The Origins and Further Development of the Jameson-Schmidt-Turkel (JST) Scheme, A. Jameson,
    AIAA Journal, Volume 55(5) May 2017,
    http://aero-comlab.stanford.edu/Papers/jst_2015_updated_07_03_2015.pdf

    Attributes:
    _k2 : float
        Second order dissipation coefficient
    _k4 : float
        Fourth order dissipation coefficient
    _c4 : float
        Fourth order multiplicative constant
    """
    def __init__(self, fluid, k2=1., k4=1/32., c4=2.):
        self._k2 = k2
        self._k4 = k4
        self._c4 = c4
        super().__init__(fluid)

    def compute_residual(self, s0, s1, ds0, ds1, n, l, d):
        # Compute central flux and average wavespeed
        s = 0.5 * (s0 + s1)
        rho, q, p = self._flu.eval_primitive(s)
        f = self._flu.compute_flux(rho, q, p, s[3])
        a = abs(q.dot(n)) + self._flu.eval_speed_sound(rho, p)

        # Compute state difference (since the normal is inward wrt cell 0, signs are reversed compared to the reference paper)
        delta_s = s0 - s1 # state difference at current edge
        delta_s0 = + ds0.dot(d * n) # state difference at "outer" edge
        delta_s1 = + ds1.dot(d * n) # state difference at "inner" edge

        # Compute shock sensor (density-based)
        sns0 = abs(delta_s[0] - delta_s0[0]) / max(abs(delta_s[0]) + abs(delta_s0[0]), 1e-8)
        sns1 = abs(delta_s1[0] - delta_s[0]) / max(abs(delta_s1[0]) + abs(delta_s[0]), 1e-8)
        sns0 *= sns0
        sns1 *= sns1

        # Compute switching functions
        eps2 = self._k2 * 0.5 * (sns0 + sns1)
        eps4 = max(0., self._k4 - self._c4 * eps2)

        # Compute JST flux
        return (f.dot(n) - a * (eps2 * delta_s - eps4 * (delta_s1 - 2 * delta_s + delta_s0))) * l
