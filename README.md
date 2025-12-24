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
- improve API
- install + cli + get models path automatically when running tests/examples
- set up CI + use pytest for test?
- rework mesh generation
- implement ROE: add MUSCL + limiters
- use autodiff: does not support array assignment, computing jacobian is not efficient (need to resort to JVP only)
- use jit: performance not any better than pure numpy, except if pure functional programming
- add doc
