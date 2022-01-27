from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import cma
import pandas as pd
import os

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

from .ES import sepCMAES, sepCEM, sepMCEM
from .models import RLNN
from collections import namedtuple
from .random_process import GaussianNoise
from .memory import Memory, Archive
from .samplers import IMSampler
from .util import *

from torchlib.deep_rl.algorithm.model_based import ModelBasedAgent
from gym_energyplus.wrappers import EnergyPlusObsWrapper


Sample = namedtuple('Sample', ('params', 'score',
                               'gens', 'start_pos', 'end_pos', 'steps'))
Theta = namedtuple('Theta', ('mu', 'cov', 'samples'))

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor
    

class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, hidden_size, mpc_horizon, layer_norm, actor_lr):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.discrete = False
        feature_dim = state_dim + action_dim
        out_features_f1 = 128
        hidden_size_lstm = 128
        hidden_size_linear = 128
        out_features_f2 = 128
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=out_features_f1)
        self.lstm = nn.LSTM(input_size=out_features_f1, hidden_size=hidden_size_lstm, num_layers=1, 
                            batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=state_dim+mpc_horizon, out_features=out_features_f2)
        self.dropout2 = nn.Dropout(0.25)  # TODO: Do I need to turn of the dropout during testing time? 
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.ReLU()
        self.linear = nn.Sequential(
            nn.Linear(hidden_size_lstm + out_features_f2, hidden_size_linear),  # NOTE: hidden_size is used twice here. 
            nn.ReLU6(),
            nn.Linear(hidden_size_linear, action_dim),
            nn.Tanh()
        )

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=actor_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    
    def forward(self, history_states, history_actions, current_state, weather_pred):
        feature = torch.cat((history_states, history_actions), dim=-1)  # (b, T, 10)

        output = self.fc1(feature)
        
        # output = self.relu1(output)
        
        output, _ = self.lstm(output) 

        output = self.dropout1(output)

        output = output[:, -1, :]
        
        input_append = torch.cat((current_state, weather_pred), dim=-1)
        input_append = self.fc2(input_append)
        input_append = self.dropout2(input_append)
        input_append = self.relu2(input_append)
        
        output = torch.cat((output, input_append), dim=-1)
        output = self.linear.forward(output)
        return output
    

    def update(self, memory, batch_size, critic):

        # Sample replay buffer
        current_state, history_obses, _, history_actions, weather_pred, _ = memory.sample(batch_size)

        # Compute actor loss
        actor_loss = -critic(current_state=current_state, 
                             history_states=history_obses,
                             current_action=self(history_obses, history_actions, current_state, weather_pred),
                             history_actions=history_actions,
                             weather_pred=weather_pred
                             ).mean()  # See Eq. (6) in the DDPG paper. 

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # NOTE: There might be some problems with retain_graph here. 
        self.optimizer.step()


class Critic(RLNN):
    
    def __init__(self, state_dim, action_dim, max_action, hidden_size, mpc_horizon, layer_norm, critic_lr):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.discrete = False
        feature_dim = state_dim + action_dim
        out_features_f1 = 128
        hidden_size_lstm = 128
        hidden_size_linear = 128
        out_features_f2 = 128
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=out_features_f1)
        self.lstm = nn.LSTM(input_size=out_features_f1, hidden_size=hidden_size_lstm, num_layers=1, 
                            batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.25)  # TODO: Maybe I can tune the dropout argument. 
        self.fc2 = nn.Linear(in_features=state_dim + mpc_horizon + action_dim, out_features=out_features_f2)
        self.dropout2 = nn.Dropout(0.25)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.ReLU()
        self.linear = nn.Sequential(
            nn.Linear(hidden_size_lstm + out_features_f2, hidden_size_linear),  # NOTE: hidden_size is used twice here. 
            nn.ReLU6(),
            nn.Linear(hidden_size_linear, 1)
        )

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=critic_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        

    def forward(self, current_state, history_states, current_action, history_actions, weather_pred):

        feature = torch.cat((history_states, history_actions), dim=-1)  # (b, T, 10)
        output = self.fc1(feature)
        
        # output = self.relu1(output)
        
        output, _ = self.lstm(output)  # (b, hidden_size)
        output = self.dropout1(output)
        output = output[:, -1, :]
        
        input_append = torch.cat((current_state, current_action, weather_pred), dim=-1)
        input_append = self.fc2(input_append)
        input_append = self.dropout2(input_append)
        input_append = self.relu2(input_append)
        
        output = torch.cat((output, input_append), dim=-1)
        output = self.linear.forward(output)
        
        return output
    

    def update(self, memory, batch_size):

        # Sample replay buffer
        current_state, history_obses, current_action, history_actions, weather_pred, score = memory.sample(batch_size)
        
        target_Q = score

        # Get current Q estimate
        current_Q = self(current_state, history_obses, current_action, history_actions, weather_pred)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optimizer.step()


class ModelBasedCEMRLAgent(ModelBasedAgent):
    def __init__(self, env, model, actor_lr, critic_lr, batch_size, mpc_horizon, window_length, 
                 hidden_size, layer_norm, temperature_center, pop_size, mem_size, gauss_sigma, 
                 sigma_init, damp, damp_limit, elitism, 
                 max_steps, start_steps, n_grad, n_noisy, n_episodes, 
                 period, n_eval, output, save_all_models):
        super(ModelBasedCEMRLAgent, self).__init__(model=model)
        self.output = output
        self.env = env
        self.env_name = self.env.__class__.__name__
        self.output = get_output_folder(self.output, self.env_name)
        with open(self.output + "/parameters.txt", 'w') as file:
            for key, value in vars(self).items():
                file.write("{} = {}\n".format(key, value))
        self.model = model
        self.actor_lr = actor_lr
        self.mem_size = mem_size
        self.pop_size = pop_size
        self.max_steps = max_steps
        self.start_steps = start_steps
        self.n_grad = n_grad
        self.batch_size = batch_size      
        self.n_noisy = n_noisy
        self.n_episodes = n_episodes
        self.period = period
        self.n_eval = n_eval
        self.save_all_models = save_all_models
        self.mpc_horizon = mpc_horizon
        self.window_length = window_length
        self.evaluate_first_call = True
        self.evaluate_first_save = True
        self.hidden_size = hidden_size
        self.temperature_center = temperature_center
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = int(self.env.action_space.high[0])
        
        outdoor_temp_data = pd.read_csv(os.getcwd() + '/outdoor_temp_extract/interpolated_outdoor_temp_Tampa.csv')
        self.outdoor_temp_data = outdoor_temp_data.values

        self.memory = Memory(memory_size=self.mem_size, state_dim=self.state_dim, action_dim=self.action_dim, 
                             window_length=self.window_length, mpc_horizon=self.mpc_horizon)
        self.critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action, hidden_size=self.hidden_size,
                             mpc_horizon=self.mpc_horizon, layer_norm=layer_norm, critic_lr=critic_lr)
        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action, hidden_size=self.hidden_size, 
                           mpc_horizon=self.mpc_horizon, layer_norm=layer_norm, actor_lr=actor_lr)
        self.a_noise = GaussianNoise(self.action_dim, sigma=gauss_sigma)

        if USE_CUDA:
            self.critic.cuda()
            self.actor.cuda()

        # CEM
        self.es = sepCEM(self.actor.get_size(), mu_init=self.actor.get_params(), sigma_init=sigma_init, damp=damp, 
                         damp_limit=damp_limit, pop_size=self.pop_size, antithetic=not self.pop_size % 2, 
                         parents=self.pop_size // 2, elitism=elitism)
        self.sampler = IMSampler(self.es)

        self.df = pd.DataFrame(columns=["total_steps", "average_score", 
                                        "average_score_rl", "average_score_ea", "best_score"])
        
        self.actor_steps = 0
        self.reused_steps = 0

        self.es_params = []
        self.fitness = []
        self.n_steps = []
        self.n_start = []

        self.old_es_params = []
        self.old_fitness = []
        self.old_n_steps = []
        self.old_n_start = []
        
    
    def evaluate(self, actor, env, model, mpc_horizon=10, memory=None, n_episodes=1, random=False, 
                 noise=None): 
        """
        Computes the score of an actor on a given number of runs,
        fills the memory if needed
        """

        if not random:
            def policy(history_obses, history_actions, current_state, weather_pred):
                history_obses = FloatTensor(history_obses)
                history_actions = FloatTensor(history_actions)
                current_state = FloatTensor(current_state)
                weather_pred = FloatTensor(weather_pred)
                action = actor(history_obses, history_actions, current_state, weather_pred).cpu().data.numpy().flatten()

                if noise is not None:
                    action += noise.sample()

                return np.clip(action, -self.max_action, self.max_action)
            
        else:
            def policy(history_obses, history_action, current_state, weather_pred):
                return env.action_space.sample()

        scores = []
        steps = 0

        for _ in range(n_episodes): # TODO: Think about why different episodes are different. 
            
            score = 0
            done = False
            step = 0
            
            self._sample_update()
            
            history_obses = FloatTensor(self.history_obses)
            history_actions = FloatTensor(self.history_actions)
            current_state = FloatTensor(self.current_state)
            weather_pred = FloatTensor(self.weather_pred)
            # get next action and act
            action = policy(history_obses, history_actions, current_state, weather_pred) 
            
            # buffer for variables to be saved in memory
            history_obses_to_memory = history_obses
            history_actions_to_memory = history_actions
            current_state_to_memory = current_state
            weather_pred_to_memory = weather_pred
            current_action_to_memory = action

            while not done:
                current_action = FloatTensor(action)
                
                history_obses = torch.cat((history_obses[:, 1:, :], torch.unsqueeze(current_state, dim=1)), dim=1)
                history_actions = torch.cat((history_actions[:, 1:, :], torch.unsqueeze(torch.unsqueeze(current_action, dim=0), dim=1)), dim=1)
                n_obs = model.predict_next_states(states=history_obses, actions=history_actions)  # TODO: The model is not trained well. 
                n_obs = torch.cat((torch.unsqueeze(weather_pred[:, 0], dim=0), n_obs), dim=1)
                raw_state = _construct_raw_state_from_obs(n_obs, env=self.env, temperature_center=self.temperature_center)
                raw_state = np.insert(arr=raw_state, obj=3, values=0, axis=0)  # insert PUE as 0, since we do not use it anyway. 
                
                if self.evaluate_first_save:
                    header = ['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
                              'WEST ZONE:Zone Air Temperature [C](TimeStep)',
                              'EAST ZONE:Zone Air Temperature [C](TimeStep)',
                              'EMS:Power Utilization Effectiveness [](TimeStep)',
                              'Whole Building:Facility Total Electric Demand Power [W](Hourly)',
                              'Whole Building:Facility Total Building Electric Demand Power [W](Hourly)',
                              'Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)'
                              ]
                    pd.DataFrame(np.expand_dims(raw_state, axis=0)).to_csv('init_mpc/raw_state.csv', header=header, index=False)
                    self.evaluate_first_save = False
                else:
                    pd.DataFrame(np.expand_dims(raw_state, axis=0)).to_csv('init_mpc/raw_state.csv', mode='a', header=False, index=False)              

                reward = env.ep_model.compute_reward(raw_state=raw_state)
                score += reward  # TODO: What about adding a discount factor?      
                step += 1
                
                if self.evaluate_first_call:
                    weather_pred = FloatTensor(self.outdoor_temp_data[self.weather_idx_ac - self.mpc_horizon + 3 + step:
                        self.weather_idx_ac + 3 + step, 1])              
                else:
                    weather_pred = FloatTensor(self.outdoor_temp_data[self.weather_idx_ac + self.window_length + 3 - 1 + step:self.weather_idx_ac 
                                                                      + self.window_length + 3 - 1 + self.mpc_horizon + step, 1])  
                # NOTE: This should be in accordance with self._sample_update()
                weather_pred = torch.unsqueeze(weather_pred, dim=0)
                
                if step == self.mpc_horizon: 
                    done = True
                    
                current_state = n_obs
                action = policy(history_obses, history_actions, current_state, weather_pred)
                
                # self.history_obses = history_obses
                # self.history_actions = history_actions
                # self.current_state = current_state  # These are for the next evaluate. TODO: Is there any problem here? 
                # # The Problem: When comparing the fitness values of different actors, I think the initial arguments should be the same. 
                # # However, the three lines above change the initial values. 

            steps += 1
            
            if self.evaluate_first_call:
                self.evaluate_first_call = False

            # adding in memory
            if memory is not None:
                memory.add((current_state_to_memory, history_obses_to_memory, current_action_to_memory, 
                            history_actions_to_memory, weather_pred_to_memory, score))

            scores.append(score)

        return np.mean(scores), steps
    

    def predict(self, history_states, history_actions, current_state, weather_pred):
        
        weather_pred = np.expand_dims(weather_pred, axis=0)
        current_state = np.expand_dims(current_state, axis=0)
        
        history_obses = FloatTensor(history_states)
        history_actions = FloatTensor(history_actions)
        current_state = FloatTensor(current_state)
        weather_pred = FloatTensor(weather_pred)
        self.actor = self.actor.eval()
        action = self.actor(history_obses, history_actions, current_state, weather_pred).cpu().data.numpy().flatten()
        self.actor = self.actor.train()
        
        return action

    
    def fit_policy(self):
           
        actions_init_mpc = pd.read_csv(os.getcwd() + '/init_mpc/actions_init_mpc.csv')
        self.actions_init_mpc = actions_init_mpc.to_numpy()
        states_init_mpc = pd.read_csv(os.getcwd() + '/init_mpc/states_init_mpc.csv')
        self.states_init_mpc = states_init_mpc.to_numpy()
        self.weather_idx_ac_backup = self.states_init_mpc.shape[0] - 60 - 960  # NOTE: 960 is 96 x 10, where 10 is the num_on_policy_rollouts. 
        
        total_steps = 0
        step_cpt = 0
        
        # training
        while total_steps < self.max_steps:

            self.fitness = np.zeros(self.pop_size)
            self.n_start = np.zeros(self.pop_size)
            self.n_steps = np.zeros(self.pop_size)
            self.es_params, n_r, idx_r = self.sampler.ask(self.pop_size, self.old_es_params)
            # n_r: Number of Reused samples 
            print("Reused {} samples".format(n_r))

            # udpate the rl actors and the critic
            if total_steps > self.start_steps:
                
                for i in range(self.n_grad):
                    # self.n_grad means the number of actors that is updated using gradient information. 

                    # set params
                    self.actor.set_params(self.es_params[i])
                    self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

                    # critic update
                    for _ in tqdm(range(int((self.actor_steps + self.reused_steps) // self.n_grad))):
                        self.critic.update(self.memory, self.batch_size)

                    # actor update
                    for _ in tqdm(range(int(self.actor_steps + self.reused_steps))):
                        self.actor.update(self.memory, self.batch_size, self.critic)

                    # get the params back in the population
                    self.es_params[i] = self.actor.get_params()

            self.actor_steps = 0
            self.reused_steps = 0

            # evaluate noisy actor(s)
            for i in range(self.n_noisy):
                self.weather_idx_ac = self.weather_idx_ac_backup
                self.actor.set_params(self.es_params[i])
                f, steps = self.evaluate(actor=self.actor, env=self.env, model=self.model, mpc_horizon=self.mpc_horizon,
                                         memory=self.memory, n_episodes=self.n_episodes, noise=self.a_noise)
                self.actor_steps += steps
                prCyan('Noisy actor {} fitness:{}'.format(i, f))

            # evaluate all actors
            for i in range(self.pop_size):
                
                self.weather_idx_ac = self.weather_idx_ac_backup

                # evaluate new actors
                if i < self.n_grad or (i >= self.n_grad and (i - self.n_grad) >= n_r):

                    self.actor.set_params(self.es_params[i])
                    pos = self.memory.get_pos()
                    f, steps = self.evaluate(actor=self.actor, env=self.env, model=self.model, mpc_horizon=self.mpc_horizon, 
                                             memory=self.memory, n_episodes=self.n_episodes)
                    self.actor_steps += steps

                    # updating arrays
                    self.fitness[i] = f
                    self.n_steps[i] = steps
                    self.n_start[i] = pos

                    # print scores
                    prLightPurple('Actor {}, fitness:{}'.format(i, f))

                # reusing actors
                else:
                    idx = idx_r[i - self.n_grad]
                    self.fitness[i] = self.old_fitness[idx]
                    self.n_steps[i] = self.old_n_steps[idx]
                    self.n_start[i] = self.old_n_start[idx]

                    # duplicating samples in buffer
                    self.memory.repeat(int(self.n_start[i]), int((self.n_start[i] + self.n_steps[i]) % self.mem_size))

                    # adding old_steps
                    self.reused_steps += self.old_n_steps[idx]

                    # print reused score
                    prGreen('Actor {}, fitness:{}'.format(i, self.fitness[i]))

            # update ea
            self.es.tell(self.es_params, self.fitness)
            
            total_steps += self.actor_steps
            step_cpt += self.actor_steps

            # update sampler stuff
            self.old_fitness = deepcopy(self.fitness)
            self.old_n_steps = deepcopy(self.n_steps)
            self.old_n_start = deepcopy(self.n_start)
            self.old_es_params = deepcopy(self.es_params)

            # save stuff
            if step_cpt >= self.period:
                
                self.weather_idx_ac = self.weather_idx_ac_backup

                # evaluate mean actor over several runs. Memory is not filled  # NOTE: Memory may be filled in my code. 
                # and steps are not counted
                self.actor.set_params(self.es.mu)
                # self._sample_update()
                f_mu, _ = self.evaluate(actor=self.actor, env=self.env, model=self.model, mpc_horizon=self.mpc_horizon,
                                        memory=None, n_episodes=self.n_eval)
                prRed('Actor Mu Average Fitness:{}'.format(f_mu))

                self.df.to_pickle(self.output + "/log.pkl")
                res = {"total_steps": total_steps,
                       "average_score": np.mean(self.fitness),
                       "average_score_half": np.mean(np.partition(self.fitness, self.pop_size // 2 - 1)[self.pop_size // 2:]),
                       "average_score_rl": np.mean(self.fitness[:self.n_grad]) if self.n_grad > 0 else None,
                       "average_score_ea": np.mean(self.fitness[self.n_grad:]),
                       "best_score": np.max(self.fitness),
                       "mu_score": f_mu,
                       "n_reused": n_r}

                if self.save_all_models:
                    os.makedirs(self.output + "/{}_steps".format(total_steps), exist_ok=True)
                    self.critic.save_model(self.output + "/{}_steps".format(total_steps), "critic")
                    self.actor.set_params(self.es.mu)
                    self.actor.save_model(self.output + "/{}_steps".format(total_steps), "actor_mu")
                else:
                    self.critic.save_model(self.output, "critic")
                    self.actor.set_params(self.es.mu)
                    self.actor.save_model(self.output, "actor")
                self.df = self.df.append(res, ignore_index=True)
                step_cpt = 0
                print(res)

            print("Total steps", total_steps)
            
            
    def _sample_update(self):

        if self.evaluate_first_call:
            history_actions = self.actions_init_mpc[self.weather_idx_ac:self.weather_idx_ac + self.window_length, :]
            history_states = self.states_init_mpc[self.weather_idx_ac:self.weather_idx_ac + self.window_length, :]
            current_state = self.states_init_mpc[self.weather_idx_ac + self.window_length, :]
            weather_pred = self.outdoor_temp_data[self.weather_idx_ac + self.window_length + 3:
                self.weather_idx_ac + self.window_length + self.mpc_horizon + 3, 1]
            self.weather_idx_ac += self.mpc_horizon + self.window_length
            
            self.sample_update_idx = 0
            
            # NOTE: The incremental step of self.weather_idx_ac can be designed manually. I set it as self.mpc_horizon out of the reason that
            # this may add the diversity of the samples in memory. 
            
        else:
            history_actions = self.actions_init_mpc[self.weather_idx_ac:self.weather_idx_ac + self.window_length, :]
            history_states = self.states_init_mpc[self.weather_idx_ac:self.weather_idx_ac + self.window_length, :]
            current_state = self.states_init_mpc[self.weather_idx_ac + self.window_length, :]
            weather_pred = self.outdoor_temp_data[self.weather_idx_ac + self.window_length + 3:self.weather_idx_ac + 
                                                  self.window_length + 3 + self.mpc_horizon, 1]
            self.weather_idx_ac += 1
                        
        self.history_obses = np.expand_dims(history_states, axis=0)
        self.history_actions = np.expand_dims(history_actions, axis=0)
        self.current_state = np.expand_dims(current_state, axis=0)
        weather_pred = np.expand_dims(weather_pred, axis=0)
        self.weather_pred = weather_pred


def _construct_raw_state_from_obs(obs, env, temperature_center):
    
    state = torch.squeeze(obs[:, :-4])
    energyplus_obs_wrapper_instance = EnergyPlusObsWrapper(env=env, temperature_center=temperature_center)
    raw_state = energyplus_obs_wrapper_instance.reverse_observation(state.detach().cpu().numpy())
    
    return raw_state
