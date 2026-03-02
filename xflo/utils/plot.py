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
import numpy as np

def save(xy, cp, mach, show=False):
    # Pressure coefficient
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axs[0].plot(xy[:, 0], cp, lw=2)
    axs[0].set_ylabel('$c_p$')
    axs[0].invert_yaxis()
    axs[1].plot(xy[:, 0], xy[:, 1], lw=2)
    axs[1].set_xlabel('$x/c$')
    axs[1].set_ylabel('$y/c$')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.savefig('cp.png')
    if show:
        plt.draw()

    # Mach number
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axs[0].plot(xy[:, 0], mach, lw=2)
    axs[0].set_ylabel('$M$')
    axs[1].plot(xy[:, 0], xy[:, 1], lw=2)
    axs[1].set_xlabel('$x/c$')
    axs[1].set_ylabel('$y/c$')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.savefig('mach.png')
    if show:
        plt.show()

    # Write data to disk
    data = np.column_stack((xy, cp, mach))
    np.savetxt('airfoil.dat', data, fmt='%8.4f', delimiter=', ', header='{:>6s}, {:>8s}, {:>8s}, {:>8s},'.format('x', 'y', 'cp', 'mach'))
