"""
Implement Twin-Delayed DDPG in Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018
The key difference with DDPG lies in
1. Add noise to target policy served as regularization to prevent overfitting to current best policy
2. Use clipped double Q function to avoid overestimation in Q value
3. Add Gaussian noise to explore at training time.
"""

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchlib import deep_rl
from torchlib.deep_rl import BaseAgent, RandomAgent
from torchlib.common import convert_to_tensor, enable_cuda, FloatTensor
from torchlib.deep_rl.utils.replay.replay import TransitionReplayBuffer
from torchlib.deep_rl.utils.replay.sampler import StepSampler
from torchlib.utils.logx import EpochLogger
from torchlib.utils.timer import Timer
from torchlib.utils.weight import soft_update, hard_update
from tqdm.auto import tqdm


class TD3Agent(BaseAgent):
    def __init__(self, nets, learning_rate, **kwargs):
        self.actor_module = nets['actor']
        self.actor_optimizer = optim.Adam(self.actor_module.parameters(), lr=learning_rate)
        self.critic_module = nets['critic']
        self.critic_optimizer = optim.Adam(self.critic_module.parameters(), lr=learning_rate)

        self.target_actor_module = copy.deepcopy(self.actor_module)
        self.target_critic_module = copy.deepcopy(self.critic_module)
        hard_update(self.target_actor_module, self.actor_module)
        hard_update(self.target_critic_module, self.critic_module)

        if enable_cuda:
            self.actor_module.cuda()
            self.critic_module.cuda()
            self.target_actor_module.cuda()
            self.target_critic_module.cuda()

    @torch.no_grad()
    def predict_batch(self, state):
        state = convert_to_tensor(state.astype(np.float32))
        return self.actor_module.forward(state).cpu().numpy()

    def state_dict(self):
        return {
            'actor': self.actor_module.state_dict(),
            'critic': self.critic_module.state_dict()
        }

    def load_state_dict(self, states):
        self.actor_module.load_state_dict(states['actor'])
        self.critic_module.load_state_dict(states['critic'])

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def update(self, replay_buffer, num_updates, action_limit, policy_freq=2, batch_size=128, target_noise=0.2,
               clip_noise=0.5, tau=5e-3, gamma=0.99):
        for i in range(num_updates):
            transition = replay_buffer.sample(batch_size)
            s_batch, a_batch, s2_batch, r_batch, t_batch = convert_to_tensor(transition, location='gpu')

            r_batch = r_batch.type(FloatTensor)
            t_batch = t_batch.type(FloatTensor)

            # get ground truth q value
            with torch.no_grad():
                target_action_noise = torch.clamp(torch.randn_like(a_batch) * target_noise, min=-clip_noise,
                                                  max=clip_noise)
                target_action = torch.clamp(self.target_actor_module.forward(s2_batch) + target_action_noise,
                                            min=-action_limit, max=action_limit)
                target_q = self.target_critic_module.forward(state=s2_batch, action=target_action, minimum=True)

                q_target = r_batch + gamma * target_q * (1 - t_batch)

            # critic loss
            q_values, q_values2 = self.critic_module.forward(s_batch, a_batch, minimum=False)
            q_values_loss = F.mse_loss(q_values, q_target) + F.mse_loss(q_values2, q_target)

            self.critic_optimizer.zero_grad()
            q_values_loss.backward()
            self.critic_optimizer.step()

            if i % policy_freq == 0:
                action = self.actor_module.forward(s_batch)
                q_values = self.critic_module.forward(s_batch, action, minimum=False)[0]
                loss = -torch.mean(q_values)
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

                soft_update(self.target_critic_module, self.critic_module, tau)
                soft_update(self.target_actor_module, self.actor_module, tau)

    def train(self, env, exp_name, actor_noise=None,
              prefill_steps=10000, num_epochs=1000, epoch_length=1000,
              replay_pool_size=1000000, replay_buffer=None,
              num_updates=10, policy_freq=2, batch_size=128,
              target_noise=0.2, clip_noise=0.5, tau=5e-3, gamma=0.99,
              log_dir=None, checkpoint_path=None, **kwargs):

        logger = EpochLogger(output_dir=log_dir, exp_name=exp_name)

        if checkpoint_path is None:
            dummy_env = env.env_fns[0]()
            checkpoint_path = os.path.join(logger.get_output_dir(), dummy_env.spec.id)
            del dummy_env

        best_mean_episode_reward = -np.inf
        timer = Timer()
        timer.reset()

        # create action noise for exploration
        if actor_noise is None:
            actor_noise = lambda: np.random.randn(env.num_envs, *env.single_action_space.shape).astype(np.float32) * 0.1

        # create replay buffer
        if replay_buffer is None:
            replay_buffer = TransitionReplayBuffer(capacity=replay_pool_size,
                                                   obs_shape=env.single_observation_space.shape,
                                                   obs_dtype=env.single_observation_space.dtype,
                                                   ac_shape=env.single_action_space.shape,
                                                   ac_dtype=env.single_action_space.dtype)

        assert np.all(env.single_action_space.high[0] == env.single_action_space.high) and \
               np.all(env.single_action_space.low[0] == env.single_action_space.low)

        action_limit = env.single_action_space.high[0]

        exploration_agent = RandomAgent(action_space=env.action_space)

        sampler = StepSampler(prefill_steps=prefill_steps, logger=logger)
        sampler.initialize(env, exploration_agent, replay_buffer)
        total_timesteps = prefill_steps // env.num_envs * prefill_steps

        exploration_agent.predict_batch = lambda state: np.clip(self.predict_batch(state) + actor_noise(),
                                                                -action_limit, action_limit)

        for epoch in range(num_epochs):
            for _ in tqdm(range(epoch_length), desc='Epoch {}/{}'.format(epoch + 1, num_epochs)):
                sampler.sample(policy=exploration_agent)

                self.update(replay_buffer, num_updates, action_limit, policy_freq, batch_size, target_noise,
                            clip_noise, tau, gamma)

            total_timesteps += epoch_length * env.num_envs
            # save best model
            avg_return = logger.get_stats('EpReward')[0]

            if avg_return > best_mean_episode_reward:
                best_mean_episode_reward = avg_return
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)

            # logging
            logger.log_tabular('Time Elapsed', timer.get_time_elapsed())
            logger.log_tabular('EpReward', with_min_and_max=True)
            logger.log_tabular('EpLength', average_only=True, with_min_and_max=True)
            logger.log_tabular('TotalSteps', total_timesteps)
            logger.log_tabular('TotalEpisodes', sampler.get_total_episode())
            logger.log_tabular('BestAvgReward', best_mean_episode_reward)
            logger.log_tabular('Replay Size', len(replay_buffer))
            logger.dump_tabular()