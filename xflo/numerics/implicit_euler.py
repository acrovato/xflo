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

from .time_integration import TimeIntegration

class ImplicitEuler(TimeIntegration):
    """Implicit (forward) Euler time integration method

    dU/dt * A - R = 0
    dU/dt * A - (J*dU + R) = 0
    (J - I * A/dt) * dU = R
    """
    def __init__(self, discretization, writer, init_cfl=1., rel_tol=1e-6, max_iter=1000, save_freq=100):
        super().__init__(discretization, writer, init_cfl, rel_tol, max_iter, save_freq)
        raise NotImplementedError('!')
