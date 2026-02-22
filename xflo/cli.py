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

def _parse_args():
    """Parse command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', help='clean workspace', action='store_true')
    parser.add_argument('file', help='python files')
    args = parser.parse_args()
    return args

def _is_editable():
    """Check if package has been installed in editable mode
    """
    from importlib.metadata import Distribution
    import json
    url = Distribution.from_name('xflo').read_text('direct_url.json')
    return json.loads(url).get('dir_info', {}).get('editable', False)

def main():
    """Entry point for xflo-run"""
    from xflo.utils.path import WorkspaceHelper
    import os, time, socket

    # Get file
    args = _parse_args()
    fname = os.path.abspath(args.file)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f'File not found: {fname}')

    # Run file
    print(f'[{socket.gethostname()} - {time.strftime("%c")}] Running:', fname)
    with WorkspaceHelper(fname, clean=args.clean):
        script = open(fname, 'r', encoding='utf-8').read()
        exec(compile(script, fname, 'exec'), {'__file__': fname, '__name__':'__main__'})

def test():
    """Entry point for xflo-test"""
    import pytest
    import os, time, socket

    # Get tests directory depending on whether package has been installed in editable mode or not
    if _is_editable():
        test_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'tests')
    else:
        test_dir = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '_testing' + os.path.sep + 'tests')

    # Run tests
    print(f'[{socket.gethostname()} - {time.strftime("%c")}] Running tests:', test_dir)
    pytest.main(['-o', 'python_files="*.py"', '-p', 'no:logging', test_dir])
