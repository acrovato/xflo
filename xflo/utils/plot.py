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

import matplotlib.pyplot as plt

def save(xy, cp, mach, show=False):
    fig, axs = plt.subplots(3, 1)

    # Pressure coefficient
    axs[0].plot(xy[:, 0], cp, lw=2)
    axs[0].set_ylabel('$c_p$')
    axs[0].invert_yaxis()

    # Mach number
    axs[1].plot(xy[:, 0], mach, lw=2)
    axs[1].set_ylabel('$M$')

    # Airfoil contour
    axs[2].plot(xy[:, 0], xy[:,1], lw=2)
    axs[2].set_xlabel('$x/c$')
    axs[2].set_ylabel('$y/c$')

    # Remove borders
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()

    # Save and show
    plt.savefig('airfoil.png')
    if show:
        plt.show()
