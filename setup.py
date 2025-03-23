# -*- coding: utf8 -*-

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

from setuptools import setup, find_packages
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('xflo/__init__.py').read(),
)[0]

setup(
    name='xflo',
    version=__version__,
    description='eXperimental Flow.',
    keywords='Euler CFD aerodynamic finite-volumes airfoils Gmsh Python',
    author='Adrien Crovato',
    author_email='',
    url='https://github.com/acrovato/xflo',
    license='GNU General Public License 3.0',
    packages=find_packages(include=['xflo*']),
    install_requires=['numpy>=1.22', 'gmsh==4.10.5'],
    classifiers=['Operating System :: OS Independent', 'Programming Language :: Python'],
)
