import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


class MCMCMethods:

    def __init__(self, target_distribution, target_distribution_grad=None):
        self.target_distribution = target_distribution
        self.target_distribution_grad = target_distribution_grad
        self.trace_interval = 10  # Update the plot every 10 samples



    def metropolis_hastings(self, init, proposal, n_samples):

        x = init()
        samples = [x]

        for _ in tqdm(range(n_samples)):
            x_proposal = proposal(x)
            acceptance_ratio = self.target_distribution(x_proposal) / self.target_distribution(x)

            if np.random.rand() < acceptance_ratio:
                x = x_proposal
            samples.append(x)

        return samples



    def gibbs_sampling(self, conditional_distributions, init, n_samples, trace=False):

        x = init()
        samples = [x]

        for _ in tqdm(range(n_samples)):
            for i, conditional_distribution in enumerate(conditional_distributions):
                x[i] = conditional_distribution(x)
            samples.append(x.copy())

        return samples



    def slice_sampling(self, init, width, n_samples, burn_in=0, thinning=1):

        x = init()
        dimension = len(x)
        samples = []

        for _ in tqdm(range(n_samples * thinning + burn_in)):
            done = False

            while not done:
                u = np.random.uniform(low=0, high=self.target_distribution(x))
                x_proposal = x - width / 2 + width * np.random.rand(dimension)

                if np.all(x_proposal < x):
                    done = True

                if self.target_distribution(x_proposal) > u:
                    x = x_proposal
                    done = True

            if _ >= burn_in:
                samples.append(x)

        return samples



    def mala(self, init, step_size, n_samples):

        if self.target_distribution_grad is None:
            raise ValueError("A gradient function for the target distribution must be provided for MALA.")

        x = init()
        samples = [x]

        for _ in tqdm(range(n_samples)):
            noise = np.random.normal(0, 1, size=x.shape)
            x_proposal = x + step_size * self.target_distribution_grad(x) / 2 + np.sqrt(step_size) * noise

            acceptance_ratio = self.target_distribution(x_proposal) * np.exp(- (x - x_proposal - step_size * self.target_distribution_grad(x_proposal) / 2)**2 / (2 * step_size)) \
                                / (self.target_distribution(x) * np.exp(- (x_proposal - x - step_size * self.target_distribution_grad(x) / 2)**2 / (2 * step_size)))

            if np.all(np.random.rand() < acceptance_ratio):
                x = x_proposal
            samples.append(x)

        return samples




    def scatter_plot(self, samples, title='', save_path=None):

        fig, ax = plt.subplots()

        scatter = ax.scatter([], [])
        ax.set_title(title)

        scatter.set_offsets(samples)

        ax.set_xlim(min([s[0] for s in samples]), max([s[0] for s in samples]))
        ax.set_ylim(min([s[1] for s in samples]), max([s[1] for s in samples]))

        if save_path:
            plt.savefig(save_path)
        plt.show()



    def trace_plot(self, samples, param_names=None, title='', save_path=None):
        samples = np.asarray(samples)

        if len(samples.shape) == 1:
            samples = samples.reshape(-1, 1)

        if param_names is None:
            param_names = [f"Param {i+1}" for i in range(samples.shape[1])]

        num_params = samples.shape[1]
        fig, axes = plt.subplots(num_params, figsize=(10, 2 * num_params), sharex=True)

        if num_params == 1:
            axes = [axes]

        for j, ax in enumerate(axes):
            ax.plot(samples[:, j])
            ax.set_title(param_names[j])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")

        plt.tight_layout()
        fig.subplots_adjust(top = 0.85)  # Adjust the top space to fit the overall title
        fig.suptitle(title, fontsize=16)

        if save_path:
            plt.savefig(save_path)
        plt.show()


