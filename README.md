# xFlo
eXperimental Flow\
Adrien Crovato, 2025.

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![Continuous integration](https://github.com/acrovato/xflo/actions/workflows/github-ci.yml/badge.svg)](https://github.com/acrovato/xflo/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Main features
xFlo is a Python code solving the compressible Euler equations around airfoils. The code is a playground dedicated to learning computational fluid dynamics and programming techniques.

## Documentation
Detailed build and use instructions can be found in the [wiki](https://github.com/acrovato/xflo/wiki).

## General todo list
- add description
- store vtk data structure in VtkWriter for faster write
- create logger
- create API
- get models path automatically when running tests/examples
- rework data structure in mesh?
- store states/primitives/residuals together?
- store states/primitives/residuals in dedicated data structure class
- move local time stepping inside Discretization?
- rework mesh generation
- implement JST and ROE
- implement implicit time integration
- use autodiff
- use jit
- use mpi
- add doc
