from copy import deepcopy
import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from .evolutionary_strategy import SepCEM
from .networks import Actor, Critic
from .random_process import GaussianNoise
from .memory import Memory
from .samplers import IMSampler
from .util import *
from torchlib.deep_rl.algorithm.model_based import ModelBasedAgent
from gym_energyplus.wrappers import EnergyPlusObsWrapper


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor
    

class ModelBasedCEMRLAgent(ModelBasedAgent):
    def __init__(self, env, model, actor_lr, critic_lr, mpc_horizon, window_length, layer_norm, 
                 temperature_center, pop_size, mem_size, gauss_sigma, sigma_init, damp, damp_limit, 
                 elitism, period, log_dir, save_all_models):
        """[summary]

        Args:
            env: EnergyPlus instance
            model: learned dynamics model
            actor_lr (float): learning rate of actor
            critic_lr (float): learning rate of critic
            mpc_horizon (int): mpc prediction horizon
            window_length (int): length of historical data that are used
            layer_norm (boolen): indicator whether to use layer normalization to the 
                                 fully connected layer of actor and critic
            temperature_center (float): center of indoor temperature that is preferred
            pop_size (int): number of actors that are considered
            mem_size (int): maximum size of memory that is used to store the evalution data 
                            and train the actors and critic
            gauss_sigma (float): standard deviation of the noise for noisy actors
            sigma_init (float): initial standard deviation of the homogeneous diagonal 
                                convariance matrix. Defaults to 1e-3.
            damp (float): initial standard deviation of the noise term adding to the
                          covariance of the CEM algorithm
            damp_limit (float): standard deviation limit of the noise term adding to the
                                covariance of the CEM algorithm
            elitism (boolen): indicator whether to always keep the elitism seen in the last round 
                              of evalution when creating new generations 
            period ([type]): [description]
            log_dir (str): logging directory
            save_all_models (boolen): [description]
        """
        super(ModelBasedCEMRLAgent, self).__init__(model=model)
        self.env = env
        self.log_dir = get_output_folder(log_dir, self.env.city)
        with open(self.log_dir + "/parameters.txt", 'w') as file:
            for key, value in vars(self).items():
                file.write("{} = {}\n".format(key, value))
        self.model = model
        self.actor_lr = actor_lr
        self.mem_size = mem_size
        self.pop_size = pop_size
        self.period = period
        self.save_all_models = save_all_models
        self.mpc_horizon = mpc_horizon
        self.window_length = window_length
        self.evaluate_first_call = True
        self.evaluate_first_save = True
        self.temperature_center = temperature_center
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high[0]
        
        outdoor_temp_data = pd.read_csv(os.getcwd() + '/outdoor_temp_extract/interpolated_outdoor_temp_{}.csv'
                                        .format(self.env.city))
        self.outdoor_temp_data = outdoor_temp_data.values

        self.memory = Memory(memory_size=self.mem_size, 
                             state_dim=state_dim, 
                             action_dim=action_dim, 
                             window_length=self.window_length, 
                             mpc_horizon=self.mpc_horizon)
        self.critic = Critic(state_dim=state_dim, 
                             action_dim=action_dim, 
                             max_action=self.max_action,
                             mpc_horizon=self.mpc_horizon, 
                             layer_norm=layer_norm, 
                             critic_lr=critic_lr)
        self.actor = Actor(state_dim=state_dim, 
                           action_dim=action_dim, 
                           max_action=self.max_action, 
                           mpc_horizon=self.mpc_horizon, 
                           layer_norm=layer_norm, 
                           actor_lr=actor_lr)
        self.a_noise = GaussianNoise(action_dim=action_dim, 
                                     sigma=gauss_sigma)

        if USE_CUDA:
            self.critic.cuda()
            self.actor.cuda()

        self.es = SepCEM(num_params=self.actor.get_size(), 
                         mu_init=self.actor.get_params(), 
                         sigma_init=sigma_init, 
                         damp=damp, 
                         damp_limit=damp_limit, 
                         pop_size=self.pop_size, 
                         antithetic=not self.pop_size % 2, 
                         parents=self.pop_size // 2, 
                         elitism=elitism)
        self.sampler = IMSampler(self.es)
        self.df = pd.DataFrame(columns=["total_steps", "average_score", "average_score_rl", 
                                        "average_score_ea", "best_score"])
        self.old_es_params = []
        
    def fit_policy(self, batch_size, n_grad, max_steps, start_steps, n_episodes, n_noisy, n_eval):
        """ Policy fitting 

        Args:
            batch_size (int): batch size of policy training
            n_grad (int): number of actors that use gradient steps to optimize
            max_steps (int): maximum number of iteration steps
            start_steps (int): step index after which the actors and critics will be updated. 
                               Before start_steps, we only evaluate the actors without updating. 
            n_episodes (int): number of evaluation episodes in each time
            n_noisy (int): number of noisy actors
            n_eval (int): number of evaluation episodes in when evaluating the actors after training
        """
           
        actions_init_mpc = pd.read_csv(os.getcwd() + '/init_mpc/actions_init_mpc.csv')
        self.actions_init_mpc = actions_init_mpc.to_numpy()
        states_init_mpc = pd.read_csv(os.getcwd() + '/init_mpc/states_init_mpc.csv')
        self.states_init_mpc = states_init_mpc.to_numpy()
        self.weather_idx_ac_backup = self.states_init_mpc.shape[0] - 60 - 960  # NOTE: 960 is 96 x 10, where 10 is the num_on_policy_rollouts. 
        
        total_steps = 0
        step_cpt = 0
        
        while total_steps < max_steps:
            fitness = np.zeros(self.pop_size)
            n_start = np.zeros(self.pop_size)
            n_steps = np.zeros(self.pop_size)
            es_params, n_r, idx_r = self.sampler.ask(self.pop_size, self.old_es_params)
            print("Reused {} samples".format(n_r))

            if total_steps > start_steps:
                for i in range(n_grad):
                    self.actor.set_params(es_params[i])
                    self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
                    
                    for _ in tqdm(range(int((actor_steps + reused_steps) // n_grad))):
                        self.critic.update(self.memory, batch_size)
                    for _ in tqdm(range(int(actor_steps + reused_steps))):
                        self.actor.update(self.memory, batch_size, self.critic)
                        
                    es_params[i] = self.actor.get_params()

            actor_steps = 0
            reused_steps = 0

            for i in range(n_noisy):
                self.weather_idx_ac = self.weather_idx_ac_backup
                self.actor.set_params(es_params[i])
                f, steps = self.evaluate(actor=self.actor, memory=self.memory, n_episodes=n_episodes, noise=self.a_noise)
                actor_steps += steps
                prCyan('Noisy actor {} fitness:{}'.format(i, f))

            for i in range(self.pop_size):
                self.weather_idx_ac = self.weather_idx_ac_backup
                if i < n_grad or (i >= n_grad and (i - n_grad) >= n_r):
                    self.actor.set_params(es_params[i])
                    pos = self.memory.get_pos()
                    f, steps = self.evaluate(actor=self.actor, memory=self.memory, n_episodes=n_episodes)
                    actor_steps += steps
                    fitness[i] = f
                    n_steps[i] = steps
                    n_start[i] = pos
                    prLightPurple('Actor {}, fitness:{}'.format(i, f))
                else:
                    idx = idx_r[i - n_grad]
                    fitness[i] = old_fitness[idx]
                    n_steps[i] = old_n_steps[idx]
                    n_start[i] = old_n_start[idx]
                    self.memory.repeat(int(n_start[i]), int((n_start[i] + n_steps[i]) % self.mem_size))
                    reused_steps += old_n_steps[idx]
                    prGreen('Actor {}, fitness:{}'.format(i, fitness[i]))

            self.es.tell(es_params, fitness)
            
            total_steps += actor_steps
            step_cpt += actor_steps
            
            old_fitness = deepcopy(fitness)
            old_n_steps = deepcopy(n_steps)
            old_n_start = deepcopy(n_start)
            self.old_es_params = deepcopy(es_params)

            if step_cpt >= self.period:
                self.weather_idx_ac = self.weather_idx_ac_backup
                self.actor.set_params(self.es.mu)
                f_mu, _ = self.evaluate(actor=self.actor, n_episodes=n_eval)
                prRed('Actor Mu Average Fitness:{}'.format(f_mu))

                self.df.to_pickle(self.log_dir + "/log.pkl")
                res = {"total_steps": total_steps,
                       "average_score": np.mean(fitness),
                       "average_score_half": np.mean(np.partition(fitness, self.pop_size // 2 - 1)[self.pop_size // 2:]),
                       "average_score_rl": np.mean(fitness[:n_grad]) if n_grad > 0 else None,
                       "average_score_ea": np.mean(fitness[n_grad:]),
                       "best_score": np.max(fitness),
                       "mu_score": f_mu,
                       "n_reused": n_r}

                if self.save_all_models:
                    os.makedirs(self.log_dir + "/{}_steps".format(total_steps), exist_ok=True)
                    self.critic.save_model(self.log_dir + "/{}_steps".format(total_steps), "critic")
                    self.actor.set_params(self.es.mu)
                    self.actor.save_model(self.log_dir + "/{}_steps".format(total_steps), "actor_mu")
                else:
                    self.critic.save_model(self.log_dir, "critic")
                    self.actor.set_params(self.es.mu)
                    self.actor.save_model(self.log_dir, "actor")
                self.df = self.df.append(res, ignore_index=True)
                step_cpt = 0
                print(res)

            print("Total steps", total_steps)
        
    def evaluate(self, actor, memory=None, n_episodes=1, random=False, noise=None): 
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
            
            action = policy(history_obses, history_actions, current_state, weather_pred) 
            
            history_obses_to_memory = history_obses
            history_actions_to_memory = history_actions
            current_state_to_memory = current_state
            weather_pred_to_memory = weather_pred
            current_action_to_memory = action

            while not done:
                current_action = FloatTensor(action)
                history_obses = torch.cat((history_obses[:, 1:, :], torch.unsqueeze(current_state, dim=1)), dim=1)
                history_actions = torch.cat((history_actions[:, 1:, :], torch.unsqueeze(torch.unsqueeze(current_action, dim=0), dim=1)), dim=1)
                n_obs = self.model.predict_next_states(states=history_obses, actions=history_actions)
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
                reward = self.env.ep_model.compute_reward(raw_state=raw_state)
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

            steps += 1
            if self.evaluate_first_call:
                self.evaluate_first_call = False
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
                
    def _sample_update(self):
        if self.evaluate_first_call:
            history_actions = self.actions_init_mpc[self.weather_idx_ac:self.weather_idx_ac + self.window_length, :]
            history_states = self.states_init_mpc[self.weather_idx_ac:self.weather_idx_ac + self.window_length, :]
            current_state = self.states_init_mpc[self.weather_idx_ac + self.window_length, :]
            weather_pred = self.outdoor_temp_data[self.weather_idx_ac + self.window_length + 3:
                self.weather_idx_ac + self.window_length + self.mpc_horizon + 3, 1]
            self.weather_idx_ac += self.mpc_horizon + self.window_length
            self.sample_update_idx = 0
            # NOTE: The incremental step of self.weather_idx_ac can be designed manually. 
            # I set it as self.mpc_horizon out of the reason that
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
    energyplus_obs_wrapper_instance = EnergyPlusObsWrapper(env=env, temperature_center=temperature_center) # TODO
    raw_state = energyplus_obs_wrapper_instance.reverse_observation(state.detach().cpu().numpy())
    return raw_state
