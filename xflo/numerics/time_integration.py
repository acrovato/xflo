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

from xflo.utils.log import logger
from xflo.utils.error import XFloNotImplemented
import numpy as np
import enum, time

class Status(enum.Enum):
    CONVERGED = 0 # has converged to prescribed tolerance
    MAX_IT = 1 # has not converged to prescribed tolerance within max. number of iteration
    FAILED = -1 # NaN in the solution/residual vector

class TimeIntegration:
    """Base class for time integrators

    Attributes:
    _disc : xflo.numerics.fv_discretization.FiniteVolume
        Spatial discretization
    _wrt : xflo.io.writer.Writer
        Data writer
    _cfl : float
        Initial CFL number
    _rtol : float
        Relative tolerance on density residual
    _mxit : int
        Maximum number of iterations
    _sfreq : int
        Interval at which to save solution to disk
    """
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
        logger.info('Setting initial condition...')
        self._disc.initialize()
        res_rho0 = np.linalg.norm(self._disc.problem.get_variables('ResidualsDensity'))
        res_rhoe0 = np.linalg.norm(self._disc.problem.get_variables('ResidualsEnergy'))
        self._wrt.write(0, self._disc.problem.get_variables())

        # Time integration
        logger.info('Starting time integration')
        logger.info('{0:>6s} {1:>8s} {2:>8s} {3:>8s} {4:>10s} {5:>10s} {6:>8s}'.format('Iter', 'CLift', 'CDrag', 'CMomY', 'Res[rho]', 'Res[rhoE]', 'CFL'))
        logger.info('{0:6d} {1:8.4f} {2:8.4f} {3:8.4f} {4:10.2f} {5:10.2f} {6:8.2f}'.format(0, self._disc.problem.get_lift_coef(), self._disc.problem.get_drag_coef(), self._disc.problem.get_pitch_coef(), 0., 0., self._cfl))
        nit = 0
        status = Status.MAX_IT
        cpu = time.perf_counter()
        while (nit < self._mxit):
            # update solution states and residuals
            self.update_solution()
            # compute relative residuals
            res_rho = np.linalg.norm(self._disc.problem.get_variables('ResidualsDensity')) / res_rho0
            res_rhoe = np.linalg.norm(self._disc.problem.get_variables('ResidualsEnergy')) / res_rhoe0
            # print status
            nit += 1
            logger.info('{0:6d} {1:8.4f} {2:8.4f} {3:8.4f} {4:10.2f} {5:10.2f} {6:8.2f}'.format(nit, self._disc.problem.get_lift_coef(), self._disc.problem.get_drag_coef(), self._disc.problem.get_pitch_coef(), np.log10(res_rho), np.log10(res_rhoe), self._cfl))
            # check convergence and save solution if required
            if res_rho <= self._rtol:
                self._wrt.write(nit, self._disc.problem.get_variables())
                status = Status.CONVERGED
                break
            elif np.isnan(res_rho):
                self._wrt.write(nit, self._disc.problem.get_variables())
                status = Status.FAILED
                break
            else:
                if nit % self._sfreq == 0:
                    self._wrt.write(nit, self._disc.problem.get_variables())
                continue
        cpu = time.perf_counter() - cpu
        logger.info(f'Computation finished! Wall-clock time: {cpu} s')
        return status

    def update_solution(self):
        """Compute solution at next iteration
        """
        raise XFloNotImplemented('TimeIntegration not implemented!')
