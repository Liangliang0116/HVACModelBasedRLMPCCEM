import numpy as np
import torch
import pandas as pd
import os
from torchlib.common import convert_numpy_to_tensor, FloatTensor
from torchlib.utils.random.sampler import BaseSampler
from torchlib.deep_rl.algorithm.model_based import BestRandomActionPlanner


class BestRandomActionHistoryPlanner(BestRandomActionPlanner):
    """
    The only difference is that the input state and action contains T time steps
    """

    def __init__(self, model, action_sampler: BaseSampler, cost_fn=None,
                 horizon=15, num_random_action_selection=4096, gamma=0.95):
        super(BestRandomActionHistoryPlanner, self).__init__(model, action_sampler, cost_fn, 
                                                             horizon, num_random_action_selection, gamma)
        outdoor_temp_data = pd.read_csv(os.getcwd() + '/outdoor_temp_extract/interpolated_outdoor_temp_Tampa.csv')
        self.outdoor_temp_data = outdoor_temp_data.values

    def predict(self, history_state, history_actions, current_state, weather_index):
        """

        Args:
            history_state: (T - 1, 6)
            history_actions: (T - 1, 4)
            current_state: (6,)

        Returns: best action (4,)

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
            return best_action
