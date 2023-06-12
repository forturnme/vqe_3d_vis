"""
read the trajectories and sample the pca space.
pca is done on trajectory['xs'].
"""
import numpy as np
from joblib import load, dump
from sklearn.decomposition import PCA
import sys
import os
import tqdm

traj_dir = 'trajectory/'
sample_dir = 'sampled/'
save = dump


def pca_sampler(sim, ham, circ, filename='trajectory.dict', n_components=2, n_samples_x=50,
                savefile='pca_sample.dict'):
    """
    do pca on xs first,
    then sample the pca space.
    save the sampled {(pca_x, pca_y): energy} to a file.
    sample is did on x and y both in range [-1, 1], with equal interval.
    x and y are the first two components of the pca and there are meshed.
    """
    # read the trajectory
    trajectory = load(traj_dir+filename)
    xs = trajectory['xs']
    ys = trajectory['ys']
    # fill PCA
    pca = PCA(n_components=n_components)
    pca.fit(xs)
    # sample the pca space
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    pca_samples = {}
    for i in tqdm.trange(n_samples_x):
        for j in range(n_samples_x):
            pca_x = -1 + 2*i/(n_samples_x-1)
            pca_y = -1 + 2*j/(n_samples_x-1)
            x = pca_x * pca.components_[0] + pca_y * pca.components_[1]
            energy = np.real(grad_ops(x)[0])[0,0]
            pca_samples[(pca_x, pca_y)] = energy
    # save the sampled pca space to a file
    save(pca_samples, sample_dir+savefile)
    return pca_samples
