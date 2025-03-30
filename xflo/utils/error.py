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

from .log import logger

class XFloError(Exception):
    """Log exception and raise
    """
    def __init__(self, message):
        logger.exception(message)
        super().__init__(message)

class XFloRuntimeError(RuntimeError):
    def __init__(self, message):
        logger.exception(message)
        super().__init__(message)

class XFloNotImplemented(NotImplementedError):
    def __init__(self, message):
        logger.exception(message)
        super().__init__(message)

class XFloFileNotFound(FileNotFoundError):
    def __init__(self, message):
        logger.exception(message)
        super().__init__(message)
