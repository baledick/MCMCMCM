
# Modular Collection of Monte Carlo Markov Chain Methods (MCMCMCM)

The `MCMCMethods` class provides an easy-to-use implementation of several popular Markov Chain Monte Carlo (MCMC) sampling algorithms, including Metropolis-Hastings, Gibbs Sampling, Slice Sampling, and Metropolis-Adjusted Langevin Algorithm (MALA). It supports tracing the sampling paths in real time or plotting them after generating the samples.

## Installation

To use this class, simply download the `mcmc_methods.py` file and include it in your project.

## Dependencies

The class requires the following Python packages:

-   NumPy
-   Matplotlib
-   tqdm

Install these packages using pip:

`pip install numpy matplotlib tqdm`

or

`pip install -r "requirements.txt"`

## Usage

1.  Import the `MCMCMethods` class:

python

`from mcmc_methods import MCMCMethods`

2.  Define your target distribution, its gradient (if applicable), proposal distribution (for Metropolis-Hastings), and initial state:

python

`def target_distribution(x):
    # ...

def target_distribution_grad(x):
    # ...

def proposal(x):
    # ...

def init():
    # ...`

3.  Create an instance of the `MCMCMethods` class:

python

`mcmc = MCMCMethods(target_distribution, target_distribution_grad)`

4.  Generate samples using the desired MCMC method:

-   Metropolis-Hastings:

python

`samples_mh = mcmc.metropolis_hastings(init, proposal, 5000)`

-   Gibbs Sampling:

python

`samples_gibbs = mcmc.gibbs_sampling(conditional_distributions, init, 5000)`

-   Slice Sampling:

python

`samples_slice = mcmc.slice_sampling(init, width=1, n_samples=5000)`

-   MALA:

python

`step_size = 0.1
samples_mala = mcmc.mala(init, step_size, 5000)`

5.  Trace the sampling path in real time or plot the results after generating the samples:

python

`mcmc.trace_plot(samples_mh, title='Metropolis-Hastings', realtime=False, subplots=True, subplot_idx=0)`

Set `realtime=True` to trace the sampling path in real time, and `realtime=False` to plot the results after generating the samples. Pass `subplots=True` and provide a `subplot_idx` parameter to display the results as subplots.

## Example


See the `examples.py` file for examples


## API Reference

### MCMCMethods class

#### `__init__(self, target_distribution, target_distribution_grad=None)`

Constructor for the `MCMCMethods` class.

-   `target_distribution` (function): The target probability distribution function to sample from.
-   `target_distribution_grad` (function, optional): The gradient of the target probability distribution function. This is required for MALA.

#### `metropolis_hastings(self, init, proposal, n_samples, burn_in=0, thinning=1)`

Performs Metropolis-Hastings sampling.

-   `init` (function): Function to generate the initial state of the Markov chain.
-   `proposal` (function): Function to generate a new proposal state given the current state.
-   `n_samples` (int): Number of samples to generate.
-   `burn_in` (int, optional): Number of initial samples to discard as burn-in. Default is 0.
-   `thinning` (int, optional): Thinning factor to reduce the autocorrelation between samples. Default is 1.

Returns: A list of generated samples.

#### `gibbs_sampling(self, conditional_distributions, init, n_samples, burn_in=0, thinning=1)`

Performs Gibbs sampling.

-   `conditional_distributions` (list of functions): List of functions representing the conditional probability distributions of each variable.
-   `init` (function): Function to generate the initial state of the Markov chain.
-   `n_samples` (int): Number of samples to generate.
-   `burn_in` (int, optional): Number of initial samples to discard as burn-in. Default is 0.
-   `thinning` (int, optional): Thinning factor to reduce the autocorrelation between samples. Default is 1.

Returns: A list of generated samples.

#### `slice_sampling(self, init, width, n_samples, burn_in=0, thinning=1, trace=False)`

Performs Slice Sampling.

-   `init` (function): Function to generate the initial state of the Markov chain.
-   `width` (float): The width of the slice interval.
-   `n_samples` (int): Number of samples to generate.
-   `burn_in` (int, optional): Number of initial samples to discard as burn-in. Default is 0.
-   `thinning` (int, optional): Thinning factor to reduce the autocorrelation between samples. Default is 1.
-   `trace` (bool, optional): If `True`, traces the sampling path in real time. Default is `False`.

Returns: A list of generated samples.

#### `mala(self, init, step_size, n_samples, burn_in=0, thinning=1)`

Performs Metropolis-Adjusted Langevin Algorithm (MALA) sampling.

-   `init` (function): Function to generate the initial state of the Markov chain.
-   `step_size` (float): Step size for the discretized Langevin dynamics.
-   `n_samples` (int): Number of samples to generate.
-   `burn_in` (int, optional): Number of initial samples to discard as burn-in. Default is 0.
-   `thinning` (int, optional): Thinning factor to reduce the autocorrelation between samples. Default is 1.

Returns: A list of generated samples.

#### `scatter_plot(self, samples, title='', save_path=None)`

Creates a scatter plot of the provided samples.

-   `samples` (list or array-like): A list or array of samples generated by one of the MCMC methods.
-   `title` (str, optional): Title for the scatter plot. Default is an empty string.
-   `save_path` (str, optional): Path to save the plot as an image file. If not provided, the plot will not be saved.

Returns: None. The function directly displays the scatter plot using `plt.show()`.

#### `trace_plot(self, samples, param_names=None, title='', save_path=None)`

Creates a trace plot of the provided samples.

-   `samples` (list or array-like): A list or array of samples generated by one of the MCMC methods.
-   `param_names` (list of str, optional): List of parameter names to be used as titles for the subplots. If not provided, default parameter names are used.
-   `title` (str, optional): Title for the entire plot. Default is an empty string.
-   `save_path` (str, optional): Path to save the plot as an image file. If not provided, the plot will not be saved.

Returns: None. The function directly displays the trace plot using `plt.show()`.