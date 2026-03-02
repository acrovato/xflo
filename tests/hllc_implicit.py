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

from xflo.utils.path import WorkspaceHelper, build_fpath
from xflo.api import init_xflo
from pytest import approx

def get_cfg():
    return {
        'Model': {
            'Name': 'naca0012',
            'AirfoilFile': build_fpath('models/naca0012_sharp.dat', __file__, recurse_lvl=1),
            'Parameters': {
                'NumCells': 50,
                'Bump': 0.2
            }
        },
        'Freestream': {
            'AoA': 2.,
            'Mach': 0.7,
            'Altitude': 0.
        },
        'Geometry': {
            'RefChord': 1.,
            'RefX': 0.25,
            'RefY': 0.
        },
        'Scheme': {
            'Name': 'HLLC'
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
                'CflInitial': 100.0,
                'CflExponent': 0.7,
                'RelativeTolerance': 1e-4,
                'MaxNumIterations': 200,
                'SaveFrequency': 10
            },
            'InnerSolver': {
                'RelativeTolerance': 1e-3,
                'AbsoluteTolerance': 1e-5,
                'MaxNumIterations': 10,
                'NumRestart': 40,
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

    # Test
    assert status.value == 0
    assert sol.get_num_iterations() == approx(38, abs=5)
    assert bdy.get_lift_coef() == approx(0.3152, abs=5e-3)
    assert bdy.get_drag_coef() == approx(0.0028, abs=5e-4)
    assert bdy.get_pitch_coef() == approx(0.0008, abs=1e-3)

def test():
    with WorkspaceHelper(__file__, clean=True):
        main()

if __name__ == '__main__':
    main()
