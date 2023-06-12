"""
plot a 3d surface plot of the sampled points
and then make a gif of the plot, with the trajectory overlaid.
all the sampled points and trajectory are read from a file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from joblib import load
import sys
import os
from PIL import Image
import imageio
from tqdm import tqdm
from sklearn.decomposition import PCA


traj_dir = 'trajectory/'
sample_dir = 'sampled/'
plot_dir = 'plot/'
gif_dir = 'plot/gif/'


def plot_sampled(samplefile='pca_sample.dict', trajfile='trajectory.dict',
                    gifname='pca_sample.gif', n_samples_x=50):
        """
        plot the sampled points and the trajectory.
        """
        # read the sampled points
        pca_samples = load(sample_dir+samplefile)
        # read the trajectory
        trajectory = load(traj_dir+trajfile)
        xs = trajectory['xs']
        ys = trajectory['ys']
        # fill PCA
        pca = PCA(n_components=2)
        pca.fit(xs)
        # plot the sampled points
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.array([p[0] for p in pca_samples.keys()])
        y = np.array([p[1] for p in pca_samples.keys()])
        z = np.array([pca_samples[p] for p in pca_samples.keys()])
        ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
        # plot the trajectory
        x = np.array([p[0] for p in xs])
        y = np.array([p[1] for p in xs])
        z = np.array(ys)
        ax.plot(x, y, z, 'r')
        # save the plot
        plt.savefig(plot_dir+gifname[:-4]+'.png')
        plt.close()
        # make a gif
        images = []
        for i in tqdm(range(100)):
            images.append(imageio.imread(plot_dir+gifname[:-4]+'.png'))
        imageio.mimsave(gif_dir+gifname, images, duration=0.1)
        return

