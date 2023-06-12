"""
given a circuit and a hamiltonian,
generate a trajectory
the trajectory is a list of parameters that
visited by the optimizer BFGS while minimizing.
and then save the trajectory list to a file.
"""
import numpy as np
from scipy.optimize import minimize
from joblib import dump, load
import sys
import os
from mindquantum import Simulator


traj_dir = 'trajectory/'
save = dump


def probe_trajectory(sim:Simulator, circ, ham, init, filename='trajectory.dict'):
    """
    minimize the energy of the circuit
    and record its parameters visited by the optimizer
    along with the energy.
    """
    # define the function to be minimized
    def func(x, grad_ops, xs, ys):
        f,g = grad_ops(x)
        f = np.real(f)[0,0]
        g = np.real(g)[0,0]
        xs.append(x)
        ys.append(f)
        return f, g

    # initialize the trajectory list
    xs = []
    ys = []
    grad_ops = sim.get_expectation_with_grad(ham, circ)

    # minimize the energy
    res = minimize(func, init, method='BFGS', jac=True, 
                   args=(grad_ops, xs, ys), options={'gtol': 1e-4,})

    # save the trajectory list to a file
    trajectory = {
        'xs': xs,
        'ys': ys
    }
    save(trajectory, traj_dir+filename)
    return trajectory

