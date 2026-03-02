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

import os, os.path

def build_fpath(fname, caller, recurse_lvl=0):
    """Build absolute file path from a target and a caller

    Parameters:
    fname : string
        Name of the target file
    caller : string
        Name of the file from which the target should be located
    recurse_lvl : int
        Number of times to recurse to parent directory
    """
    recurse = ''
    for _ in range(recurse_lvl):
        recurse = os.path.join(recurse, '..')
    return os.path.abspath(os.path.join(os.path.dirname(caller), recurse, fname))

class WorkspaceHelper:
    """Create and switch to workspace directory"""
    def __init__(self, fname, clean=False, verb=True):
        # Get current directory and save it
        self._ori_dir = os.path.abspath(os.getcwd()) + os.path.sep

        # Build the name of the workspace folder
        if verb: print(f'Setting run directory for "{fname}"')
        common = os.path.commonprefix((fname, self._ori_dir))
        resdir = fname[len(common):].replace(os.path.sep, '_')
        resdir = os.path.splitext(resdir)[0] # remove ".py"
        self._new_dir = os.path.join('workspace', resdir)

        # Create the directory and clean it
        if not os.path.isdir(self._new_dir):
            if verb: print('- creating', self._new_dir)
            os.makedirs(self._new_dir)
        elif os.path.isdir(self._new_dir) and clean:
            if verb: print('- cleaning', self._new_dir)
            import shutil
            for f in os.listdir(self._new_dir):
                fpth = os.path.join(self._new_dir, f)
                if os.path.isfile(fpth):
                    os.remove(fpth)
                elif os.path.isdir(fpth):
                    shutil.rmtree(fpth)

    def __enter__(self):
        os.chdir(self._new_dir)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        os.chdir(self._ori_dir)
