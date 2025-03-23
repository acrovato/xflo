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

import numpy as np
import enum, time

class Status(enum.Enum):
    CONVERGED = 0 # has converged to prescribed tolerance
    MAX_IT = 1 # has not converged to prescribed tolerance within max. number of iteration
    FAILED = -1 # NaN in the solution/residual vector

class TimeIntegration:
    def __init__(self, discretization, writer, init_cfl, rel_tol, max_iter, save_freq):
        self._disc = discretization
        self._wrt = writer
        self._cfl = init_cfl
        self._rtol = rel_tol
        self._mxit = max_iter
        self._sfreq = save_freq

    def run(self):
        """Perform time integration

        Returns:
        status : xflo.Status enum
            Solver status
        """
        # Set initial condition
        print('Setting initial condition...', end=' ')
        self._disc.initialize()
        self._disc.compute_residuals()
        res_rho0 = np.linalg.norm(self._disc.get_residuals()[::4])
        res_rhoe0 = np.linalg.norm(self._disc.get_residuals()[3::4])
        print('done.')

        # Time integration
        print('Starting time integration')
        print('{0:>12s} {1:>12s} {2:>12s} {3:>12s}'.format('Iter', 'Res[rho]', 'Res[rhoE]', 'CFL'))
        nit = 0
        status = Status.MAX_IT
        cpu = time.perf_counter()
        while (nit < self._mxit):
            # update solution states and residuals
            self.compute_step()
            # compute relative residuals
            res_rho = np.linalg.norm(self._disc.get_residuals()[::4]) / res_rho0
            res_rhoe = np.linalg.norm(self._disc.get_residuals()[3::4]) / res_rhoe0
            # print status
            print('{0:12d} {1:12.6f} {2:12.2f} {3:12.2f}'.format(nit, np.log10(res_rho), np.log10(res_rhoe), self._cfl))
            # save solution
            if nit % self._sfreq == 0:
                self._wrt.write(nit, self._disc.problem.get_solution())
            nit += 1
            # check convergence
            if res_rho <= self._rtol:
                status = Status.CONVERGED
                break
            elif np.isnan(res_rho):
                status = Status.FAILED
                break
            else:
                continue
        cpu = time.perf_counter() - cpu
        print(f'Computation done! Wall-clock time: {cpu} s')
        return status

    def compute_step(self):
        """Compute solution u at next iteration
        """
        raise NotImplementedError('TimeIntegration.compute_step: not implemented!')

    def _compute_local_timestep(self):
        """Compute local time step divided by cell area
        TODO move inside disc?

        Returns:
        dt : np.array(float)
            Local time step of cells
        """
        area = self._disc.problem.mesh.get_cells_area()
        _, u, v, _, mach = self._disc.problem.get_variables()
        dt_a = np.zeros(area.shape[0], dtype=float)
        for i_cell in range(area.shape[0]):
            q = np.array([u[i_cell], v[i_cell]])
            a = np.linalg.norm(q) + np.linalg.norm(q) / mach[i_cell]
            dt_a[i_cell] = self._cfl * np.sqrt(area[i_cell]) / a / area[i_cell]
        return np.repeat(dt_a, 4)
