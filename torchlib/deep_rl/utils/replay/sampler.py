"""
Sample class for various RL algorithms. Typical RL algorithms contains two parts:
1. Sample trajectories or transitions and add to the buffer
2. Use the buffer to update the policy or value functions
"""

import gym
import numpy as np


class Sampler(object):
    def __init__(self):
        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        assert isinstance(env, gym.vector.VectorEnv), \
            'env must be Vector env for consistency. Use SynchronizeEnv to wrap for num_env=1'
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self, policy=None):
        raise NotImplementedError

    def sample_trajectories(self, policy=None):
        raise NotImplementedError

    def close(self):
        self.env.close()


class StepSampler(Sampler):
    """
    Make a sample per update
    """

    def __init__(self, prefill_steps, logger=None):
        super(StepSampler, self).__init__()
        self.current_observation = None
        self.logger = logger
        self.prefill_steps = prefill_steps

    def initialize(self, env, policy, pool):
        super(StepSampler, self).initialize(env=env, policy=policy, pool=pool)
        self.current_observation = self.env.reset()
        self.total_episodes = 0
        self.ep_rewards = np.zeros(shape=(self.env.num_envs))
        self.ep_length = np.zeros(shape=(self.env.num_envs), dtype=np.int)
        for _ in range(self.prefill_steps // self.env.num_envs):
            self.sample()

    def sample(self, policy=None):
        policy = self.policy if policy is None else policy
        action = policy.predict_batch(self.current_observation)
        next_observation, reward, terminal, info = self.env.step(action)

        self.ep_rewards += reward
        self.ep_length += 1

        for i in range(terminal.shape[0]):
            if terminal[i] == True:
                self.logger.store(EpReward=self.ep_rewards[i])
                self.ep_rewards[i] = 0.
                self.logger.store(EpLength=self.ep_length[i])
                self.ep_length[i] = 0
                self.total_episodes += 1

        self.pool.add_batch(
            states=self.current_observation,
            actions=action,
            next_states=next_observation,
            rewards=reward,
            dones=terminal)

        self.current_observation = next_observation

    def get_total_episode(self):
        return self.total_episodes