#!/usr/bin/python3
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

from xflo.api import init_xflo
from xflo.utils import path, plot

def get_cfg():
    return {
        'Model': {
            'Name': 'rae2822',
            'AirfoilFile': path.build_fpath('models/rae2822.dat', __file__, recurse_lvl=1),
            'Parameters': {
                'NumCells': 50,
                'Bump': 0.2
            }
        },
        'Freestream': {
            'AoA': 2.8,
            'Mach': 0.73,
            'Altitude': 0.
        },
        'Geometry': {
            'RefChord': 1.,
            'RefX': 0.25,
            'RefY': 0.
        },
        'Scheme': {
            'Name': 'Roe',
            'Parameters': {
                'EntropyFix': 5e-1
            }
        },
        'Gradient': {
            'Name': 'GreenGauss',
            'Limiter': {
                'Name': 'Venkatakrishnan',
                'Parameters': {
                    'Coefficient': 5.0
                }
            }
        },
        'TimeIntegration': {
            'Name': 'ImplicitEuler',
            'Parameters': {
                'CflInitial': 5.0,
                'CflExponent': 0.7,
                'RelativeTolerance': 1e-6,
                'MaxNumIterations': 200,
                'SaveFrequency': 10
            },
            'InnerSolver': {
                'RelativeTolerance': 1e-3,
                'AbsoluteTolerance': 1e-5,
                'MaxNumIterations': 20,
                'NumRestart': 20,
                'DropTolerance': 1e-6,
                'FillFactor': 20,
            }
        }
    }

def main():
    # Initialize XFlo
    xflo = init_xflo(get_cfg())
    bdy, sol = (xflo.get(key) for key in ['Body', 'Solver'])

    # Run time integration
    status = sol.run()

    # Plot results
    xy = bdy.get_coords(True)
    cp = bdy.get_pressure_coef(True)
    mach = bdy.get_mach(True)
    plot.save(xy, cp, mach, show=True)

if __name__ == '__main__':
    main()
