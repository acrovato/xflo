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

import os, sys, logging

class LoggerInterface():
    """Collection of loggers

    Attributes:
    _loggers : dict(str, logging.Logger)
        Map between a logger and its name
    """
    def __init__(self):
        self._loggers = {}

    def _initialize(self, logger_name):
        """Create logger

        Parameters:
        logger_name : str
            Name of logger assigned to current directory
        """
        # Create a logger for the current directory
        logger = logging.getLogger(logger_name)

        # Add a file handler
        log_file = logging.FileHandler(filename='log_xflo.txt', mode='w')
        log_file.setLevel(logging.INFO)
        log_file.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s: %(message)s'))
        logger.addHandler(log_file)

        # Assign the logger to current directory
        self._loggers[logger_name] = logger

    def _get_logger_name(self):
        """Returns:
        Name of logger for current directory : str"""
        return os.path.basename(os.getcwd()) + '_xflo'

    def get(self):
        """Returns:
        Logger for current directory : logging.Logger"""
        lname = self._get_logger_name()
        if lname not in self._loggers:
            self._initialize(lname)
        return self._loggers[lname]

# Create common loggin setup for console output
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)-8s: %(message)s',
                    datefmt='%m-%d-%H:%M:%S',
                    stream=sys.stdout)
# Create interface
logger_interface = LoggerInterface()
