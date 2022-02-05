import numpy as np
import torch
import pandas as pd
import os

from torchlib.common import convert_numpy_to_tensor, FloatTensor
from torchlib.utils.random.sampler import BaseSampler
from torchlib.deep_rl.algorithm.model_based import BestRandomActionPlanner


class BestRandomActionHistoryPlanner(BestRandomActionPlanner):
    """
    The random shooting algorithm
    """

    def __init__(self, 
                 model, 
                 action_sampler: BaseSampler, 
                 cost_fn=None, 
                 city=None,
                 horizon=15, 
                 num_random_action_selection=4096, 
                 gamma=0.95):
        """ Initialize this class to get an instance. 

        Args:
            model: learned dynamics model
            action_sampler (BaseSampler): sampler to generate actions
            cost_fn (optional): cost function to determine which action sequence is the best. Defaults to None.
            city (str, optional): city of which the weather file is used. Defaults to None.
            horizon (int, optional): mpc prediction horizon. Defaults to 15.
            num_random_action_selection (int, optional): number of random actions to be selected. Defaults to 4096.
            gamma (float, optional): discount factor for future cost. Defaults to 0.95.
        """
        super(BestRandomActionHistoryPlanner, self).__init__(model, action_sampler, cost_fn,
                                                             horizon, num_random_action_selection, gamma)
        outdoor_temp_data = pd.read_csv(os.getcwd() + '/outdoor_temp_extract/interpolated_outdoor_temp_{}.csv'.format(city))
        self.outdoor_temp_data = outdoor_temp_data.values

    def predict(self, history_state, history_actions, current_state, weather_index):
        """ Obtain the best action sequence with random shooting algorithm

        Args:
            history_state (np.ndarray): historical state sequence
            history_actions (np.ndarray): historical action sequence
            current_state (np.ndarray): current state vector
            weather_index (int): weather index to determine what weather information to be used

        Returns:
            np.ndarray: the best action selected by random shooting algorithm
        """
        states = np.expand_dims(history_state, axis=0)
        states = np.tile(states, (self.num_random_action_selection, 1, 1))
        states = convert_numpy_to_tensor(states)

        next_states = np.expand_dims(current_state, axis=0)
        next_states = np.tile(next_states, (self.num_random_action_selection, 1))
        next_states = convert_numpy_to_tensor(next_states)

        actions = self.action_sampler.sample((self.horizon, self.num_random_action_selection)) 
        actions = convert_numpy_to_tensor(actions)

        history_actions = np.expand_dims(history_actions, axis=0)
        current_action = np.tile(history_actions, (self.num_random_action_selection, 1, 1))
        current_action = convert_numpy_to_tensor(current_action)

        self.model.eval()
        with torch.no_grad():
            cost = torch.zeros(size=(self.num_random_action_selection,)).type(FloatTensor)
            for i in range(self.horizon):
                states = torch.cat((states, torch.unsqueeze(next_states, dim=1)), dim=1)
                current_action = torch.cat((current_action, torch.unsqueeze(actions[i], dim=1)), dim=1)
                next_states = self.model.predict_next_states(states, current_action)
                next_states_numpy = next_states.cpu().detach().numpy()
                next_states_numpy = np.insert(next_states_numpy, obj=0, values=self.outdoor_temp_data[weather_index + i, 1], axis=1)
                next_states = convert_numpy_to_tensor(next_states_numpy)
                cost += self.cost_fn(states[:, -1, :], actions[i], next_states) * self.gamma_inverse
                current_action = current_action[:, 1:, :]
                states = states[:, 1:, :]

            best_action = actions[0, torch.argmin(cost, dim=0)]
            best_action = best_action.cpu().numpy()
        
        self.model.train()    
        
        return best_action
    
    
class BestRandomActionHistoryAdaptivePlanner(BestRandomActionPlanner):
    """
    The random shooting algorithm
    """

    def __init__(self, 
                 model, 
                 action_space,
                 action_sampler: BaseSampler, 
                 cost_fn=None, 
                 city=None,
                 horizon=15, 
                 num_random_action_selection=4096, 
                 ratio_elite=0.1,
                 gamma=0.95,
                 damp=1e-3,
                 damp_limit=1e-5):
        """ Initialize this class to get an instance. 

        Args:
            model: learned dynamics model
            action_sampler (BaseSampler): sampler to generate actions
            cost_fn (optional): cost function to determine which action sequence is the best. Defaults to None.
            city (str, optional): city of which the weather file is used. Defaults to None.
            horizon (int, optional): mpc prediction horizon. Defaults to 15.
            num_random_action_selection (int, optional): number of random actions to be selected. Defaults to 4096.
            gamma (float, optional): discount factor for future cost. Defaults to 0.95.
        """
        super(BestRandomActionHistoryAdaptivePlanner, self).__init__(model, action_sampler, cost_fn, 
                                                                     horizon, num_random_action_selection, gamma)
        outdoor_temp_data = pd.read_csv(os.getcwd() + '/outdoor_temp_extract/interpolated_outdoor_temp_{}.csv'.format(city))
        self.outdoor_temp_data = outdoor_temp_data.values
        self.max_actions = action_space.high
        self.min_actions = action_space.low
        self.action_dim = action_space.shape[0]
        
        self.damp = damp
        self.damp_init = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
    
        self.num_iterations = 4
        self.samples_per_iteration = int(num_random_action_selection / self.num_iterations)
        self.num_elites = int(self.samples_per_iteration * ratio_elite)
        
        # self.weights = np.array([np.log((self.num_elites + 1) / i)
        #                          for i in range(1, self.num_elites + 1)])
        # self.weights /= self.weights.sum()
        # self.weights = np.float32(self.weights)
        
    def reset_damp(self):
        self.damp = self.damp_init

    def predict(self, history_state, history_actions, current_state, weather_index):
        """ Obtain the best action sequence with random shooting algorithm

        Args:
            history_state (np.ndarray): historical state sequence
            history_actions (np.ndarray): historical action sequence
            current_state (np.ndarray): current state vector
            weather_index (int): weather index to determine what weather information to be used

        Returns:
            np.ndarray: the best action selected by random shooting algorithm
        """
        
        self.model.eval()
        
        for _ in range(self.num_iterations):
            
            states = np.expand_dims(history_state, axis=0)
            states = np.tile(states, (self.samples_per_iteration, 1, 1))
            states = convert_numpy_to_tensor(states)

            next_states = np.expand_dims(current_state, axis=0)
            next_states = np.tile(next_states, (self.samples_per_iteration, 1))
            next_states = convert_numpy_to_tensor(next_states)
            
            history_actions_ = np.expand_dims(history_actions, axis=0)
            current_action = np.tile(history_actions_, (self.samples_per_iteration, 1, 1))
            current_action = convert_numpy_to_tensor(current_action)
        
            actions = self.action_sampler.sample((self.horizon, self.samples_per_iteration))
            actions = np.clip(actions, self.min_actions, self.max_actions)
            actions = convert_numpy_to_tensor(actions).type(FloatTensor)

            with torch.no_grad():
                cost = torch.zeros(size=(self.samples_per_iteration,)).type(FloatTensor)
                for i in range(self.horizon):
                    states = torch.cat((states, torch.unsqueeze(next_states, dim=1)), dim=1)
                    current_action = torch.cat((current_action, torch.unsqueeze(actions[i], dim=1)), dim=1)
                    next_states = self.model.predict_next_states(states, current_action)
                    next_states_numpy = next_states.cpu().detach().numpy()
                    next_states_numpy = np.insert(next_states_numpy, obj=0, values=self.outdoor_temp_data[weather_index + i, 1], axis=1)
                    next_states = convert_numpy_to_tensor(next_states_numpy)
                    cost += self.cost_fn(states[:, -1, :], actions[i], next_states) * self.gamma_inverse
                    current_action = current_action[:, 1:, :]
                    states = states[:, 1:, :]
                    
            # cost = cost.cpu().numpy()
            # idx_sorted = np.argsort(cost)
            # elites = actions[0, idx_sorted[:self.num_elites]]
            # elites = elites.cpu().numpy()

            # old_mu = self.action_sampler.mu
            # self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
            # self.action_sampler.mu = self.weights @ elites

            # z = np.float32(elites - old_mu)
            # self.action_sampler.sigma = self.weights @ (z * z) + self.damp * np.ones(self.action_dim)

            idxes_elites = torch.topk(input=-cost, k=self.num_elites, dim=0)[1].tolist()
            elites = actions[0, idxes_elites]
            elites = elites.cpu().numpy()
            elites_mean = np.mean(elites, axis=0)
            elites_std = np.std(elites, axis=0)
            
            self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
            self.action_sampler.mu = elites_mean
            self.action_sampler.sigma = elites_std + self.damp * np.ones(self.action_dim)
                
        self.action_sampler.reset_params()
        self.reset_damp()
            
        self.model.train()  
        best_action = actions[0, idxes_elites[0]].cpu().numpy()
        
        return best_action
