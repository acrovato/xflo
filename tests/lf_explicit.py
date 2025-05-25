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

def get_model(name):
    import os.path
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', name)

def get_cfg():
    return {
        'Model': {
            'Name': 'naca0012',
            'AirfoilFile': get_model('naca0012_sharp.dat')
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
            'Name': 'LaxFriedrichs',
        },
        'TimeIntegration': {
            'Name': 'ExplicitEuler',
            'Parameters': {
                'CflInitial': 0.25,
                'RelativeTolerance': 1e-2,
                'MaxNoIterations': 2000,
                'SaveFrequency': 100
            }
        }
    }

def main():
    # Initialize XFlo
    xflo = init_xflo(get_cfg())
    bdy, sol = (xflo.get(key) for key in ['Body', 'Solver'])

    # Run time integration
    status = sol.run()

    # TODO add tests
    assert status.value == 0
    assert bdy.get_lift_coef() == 0.1601
    assert bdy.get_drag_coef() == 0.0739
    assert bdy.get_pitch_coef() == -0.0222

if __name__ == '__main__':
    main()
