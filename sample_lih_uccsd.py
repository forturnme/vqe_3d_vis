"""
first, get trajectory of the uccsd circuit for lih.
then, do pca on the trajectory.
then, sample the pca space.
finally, plot the sampled points and the trajectory.
"""
import numpy as np
from scipy.optimize import minimize
from pca_sampler import pca_sampler
from probe_trajectory import probe_trajectory
from plot_sampled import plot_sampled
from mindquantum import Simulator, Hamiltonian, Circuit, UN, H
from mindquantum.algorithm.nisq import generate_uccsd
from openfermion import MolecularData

# define the geometry of the molecule
geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.4))]
# define the basis set
basis = 'sto-3g'
# define the multiplicity
multiplicity = 1
# define the charge
charge = 0
# define the bond length
bond_len = 1.4
# define the molecule
molecule = MolecularData(geometry, basis, multiplicity, charge, description=str(bond_len))
# load the molecule
molecule.load()
# get the hamiltonian and circuit
circ, _, ham, nq, ne = generate_uccsd(molecule)
# define the simulator
sim = Simulator('mqvector' ,nq)
# define the initial parameters
init = np.random.rand(circ.n_params)
# probe the trajectory
probe_trajectory(sim, circ, ham, init, filename='lih_uccsd.dict')
# do pca on the trajectory
pca_samples = pca_sampler(sim, ham, circ, filename='lih_uccsd.dict', n_components=2, n_samples_x=50,
                savefile='lih_uccsd_pca_sample.dict')
# plot the sampled points and the trajectory
plot_sampled(samplefile='lih_uccsd_pca_sample.dict', trajfile='lih_uccsd.dict',
                gifname='lih_uccsd_pca_sample.gif', n_samples_x=50)
