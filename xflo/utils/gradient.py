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

import numpy as np

def compute_grad_fd(fun, args, wrt_args, eps=1e-6):
    """Compute gradient of a function using finite differences

    Parameters:
    fun : function
        Function to differentiate
    args : list(np.array(float))
        Arguments of function
    wrt_args : list(int)
        Inidices of arguments with respect to which function will be differentiated
    eps : float
        Finite difference step

    Returns:
    grads : list(np.array(float))
        Gradients of function with respect to arguments
    """
    # Compute reference value
    f = fun(*args)
    nf = f.shape[0]
    # Compute gradients
    grads = []
    for i_arg, arg in enumerate(wrt_args):
        nv = args[arg].shape[0]
        grads.append(np.zeros((nf, nv), dtype=float))
        for i_var in range(nv):
            # perturb variable
            v = args[arg][i_var]
            args[arg][i_var] += eps
            # compute gradient
            grads[i_arg][:, i_var] = (fun(*args) - f) / eps
            # reset variable
            args[arg][i_var] = v
    return grads
