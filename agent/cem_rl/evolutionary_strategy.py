import numpy as np
from copy import deepcopy

from .optimizers import Adam, BasicSGD


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))]
    which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class SepCEM:
    """
    Cross-entropy methods.
    """

    def __init__(self, 
                 num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False):
        """ Create an instance of SepCEM. 

        Args:
            num_params (int): number of parameters subject to update of CEM
            mu_init (np.ndarray, optional): initial mean vector. Defaults to None.
            sigma_init (float, optional): initial standard deviation of the homogeneous diagonal 
                                          convariance matrix. Defaults to 1e-3.
            pop_size (int, optional): population size. Defaults to 256.
            damp (float, optional): initial standard deviation of the noise term adding to the
                                    covariance of the CEM algorithm. Defaults to 1e-3.
            damp_limit (float, optional): standard deviation limit of the noise term adding to the
                                          covariance of the CEM algorithm. Defaults to 1e-5.
            parents (int, optional): number of populations that are used to create the next generation. 
                                     Defaults to None.
            elitism (bool, optional): indicator whether to always keep the elitism seen in the last round 
                                      of evalution when creating new generations. Defaults to False.
            antithetic (bool, optional): If true, use antithetic rule to create new generations. 
                                         See method "ask" for further details. Defaults to False. 
        """

        self.num_params = num_params

        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """ generate a certain number of populations

        Args:
            pop_size (int): population size to be generated

        Returns:
            list: a list of new individuals
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite
        return inds

    def tell(self, solutions, scores):
        """ Update the mean and covariance of the individual generating distribution
        based on the fitness values in the evalution. 

        Args:
            solutions (np.ndarray): candidate solutions (e.g., neural network parameters) 
                                    which are used to update the distribution parameters
            scores (np.ndarray): scores of candidate solutions which describe how good the solutions are
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = solutions[idx_sorted[:self.parents]] - old_mu
        self.cov = 1 / self.parents * self.weights @ (z * z) \
            + self.damp * np.ones(self.num_params)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)
