import numpy as np
import pandas as pd
import os
from collections import deque

from torchlib.deep_rl import BaseAgent
from agent.utils import EpisodicHistoryDataset
    
    
class Sampler:
    """
    The sampler to get samples from the environment. 
    
    """
    def __init__(self, env, window_length, mpc_horizon):
        self.env = env
        self.window_length = window_length
        self.mpc_horizon = mpc_horizon
        
        self.history_states = deque(maxlen=self.window_length)
        self.history_actions = deque(maxlen=self.window_length)
        self.current_state = deque(maxlen=1)
        
        self.weather_index = 2
        self.create_header_when_save = True

        outdoor_temp_data = pd.read_csv(os.getcwd() + '/outdoor_temp_extract/interpolated_outdoor_temp_{}.csv'
                                        .format(self.env.city))
        self.outdoor_temp_data = outdoor_temp_data.values
        

    def sample(self, policy: BaseAgent, num_rollouts, max_rollout_length):
        
        dataset = EpisodicHistoryDataset(window_length=self.window_length)

        for _ in range(num_rollouts):
            state = self.env.reset()
            done = False
            t = 0
            while not done:
                if state.dtype == np.float:
                    state = state.astype(np.float32)

                if policy.__class__.__name__ == 'RandomAgent': 
                    action = policy.predict(state)
                elif policy.__class__.__name__ == 'ModelBasedHistoryDaggerAgent':
                    action = policy.predict(state, self.weather_index)
                elif policy.__class__.__name__ == 'ModelBasedHistoryPlanAgent':
                    action = policy.predict(state, self.weather_index)
                else:
                    history_states = np.expand_dims(np.array(list(self.history_states)), axis=0).astype(np.float32)
                    history_actions = np.expand_dims(np.array(list(self.history_actions)), axis=0).astype(np.float32)
                    weather_pred = self.outdoor_temp_data[self.weather_index: self.weather_index + self.mpc_horizon, 1]
                
                    # NOTE: Why do we need weather_index here? Because we need weather prediction! 
                    action = policy.predict(history_states=history_states, history_actions=history_actions, 
                                            current_state=state, weather_pred=weather_pred)  # TODO: The arguments need to be constructed. 
                
                self.history_states.append(state)
                self.history_actions.append(action)
                
                if 'states_init_mpc' in locals():
                    states_init_mpc = np.concatenate((states_init_mpc, np.expand_dims(state, axis=0)), axis=0)
                    actions_init_mpc = np.concatenate((actions_init_mpc, np.expand_dims(action, axis=0)), axis=0)
                else:
                    states_init_mpc = np.expand_dims(state, axis=0)
                    actions_init_mpc = np.expand_dims(action, axis=0)          

                if isinstance(action, np.ndarray) and action.dtype == np.float:
                    action = action.astype(np.float32)

                next_state, reward, done, _ = self.env.step(action)
                self.weather_index += 1

                if next_state.dtype == np.float:
                    next_state = next_state.astype(np.float32)

                done = done or (t >= max_rollout_length)

                dataset.add(state, action, next_state, reward, done)
                
                state = next_state
                t += 1
                
        if self.create_header_when_save: 
            pd.DataFrame(states_init_mpc).to_csv('init_mpc/states_init_mpc.csv', header=False, index=False)
            pd.DataFrame(actions_init_mpc).to_csv('init_mpc/actions_init_mpc.csv', header=False, index=False)
            self.create_header_when_save = False
        else:
            pd.DataFrame(states_init_mpc).to_csv('init_mpc/states_init_mpc.csv', mode='a', header=False, index=False)
            pd.DataFrame(actions_init_mpc).to_csv('init_mpc/actions_init_mpc.csv', mode='a', header=False, index=False)

        return dataset
