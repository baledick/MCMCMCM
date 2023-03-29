import numpy as np
import matplotlib.pyplot as plt
from mcmcmethods import MCMCMethods

# Define your target distribution, its gradient, proposal distribution, and initial state
def target_distribution(x):
    return np.exp(-(x**2).sum() / 2)

def target_distribution_grad(x):
    return -x

def proposal(x):
    return x + np.random.normal(0, 1, size=x.shape)

def init():
    return np.random.normal(size=2)
    
# Define the conditional distributions
def conditional_distribution_1(x):
    return np.random.normal(x[1], 1)

def conditional_distribution_2(x):
    return np.random.normal(x[0], 1)

conditional_distributions = [conditional_distribution_1, conditional_distribution_2]

## Initialising the class and starting the examples

mcmc = MCMCMethods(target_distribution, target_distribution_grad)

# Metropolis-Hastings
samples_mh = mcmc.metropolis_hastings(init, proposal, n_samples = 500)

mcmc.scatter_plot(samples_mh, title='Metropolis-Hastings')

mcmc.trace_plot(samples_mh, title='Metropolis-Hastings')


# Gibbs Sampling
conditional_distributions = [lambda x: np.random.normal(x[1], 1), lambda x: np.random.normal(x[0], 1)]

samples_gibbs = mcmc.gibbs_sampling(conditional_distributions, init, n_samples = 500)

mcmc.scatter_plot(samples_gibbs, title='Gibbs Sampling')

mcmc.trace_plot(samples_gibbs, title='Gibbs Sampling')


# # Slice Sampling
samples_slice = mcmc.slice_sampling(init, width=1, n_samples=500)

mcmc.scatter_plot(samples_slice, title='Slice Sampling')

mcmc.trace_plot(samples_slice, title='Slice Sampling')


# MALA
samples_mala = mcmc.mala(init, step_size = 0.1, n_samples = 500)

mcmc.scatter_plot(samples_mala, title='MALA')

mcmc.trace_plot(samples_mala, title='MALA')


# Hamiltonian Monte Carlo (HMC)
samples_hmc = mcmc.hmc(init, step_size = 0.1, num_steps = 10, n_samples=500)

mcmc.scatter_plot(samples_hmc, title='Hamiltonian Monte Carlo (HMC)')

mcmc.trace_plot(samples_hmc, title='Hamiltonian Monte Carlo (HMC)')


# Metropolis-Within-Gibbs
conditional_distributions = [lambda x: np.random.normal(x[1], 1), lambda x: np.random.normal(x[0], 1)]

proposal_distributions = [proposal, proposal]

samples_metropolis_within_gibbs = mcmc.metropolis_within_gibbs(init, conditional_distributions, proposal_distributions, n_samples=500)

mcmc.scatter_plot(samples_metropolis_within_gibbs, title='Metropolis-Within-Gibbs')

mcmc.trace_plot(samples_metropolis_within_gibbs, title='Metropolis-Within-Gibbs')
