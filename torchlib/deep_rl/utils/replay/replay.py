"""
Replay buffer for all RL algorithms including
* On policy: PPO
    - Collect multiple trajectories and use these to update the policy
    - Clear the buffer and recollect the data
    - Don't need to support multiple states window since it is on-policy.
* Off policy: DQN, TD3, SAC
    - Collect on-policy data n times
    - Sample a batch for update
    - Need to support multiple states window for POMDP. Must store the trajectories instead of (s, a, s', r, d) tuple
* Model-based RL
    - Collect several trajectories or transitions
    - Use those transitions to learn a model
    - Need to support multiple states window for POMDP. Must store the trajectories instead of (s, a, s', r, d) tuple
"""

from collections import namedtuple, deque

import numpy as np
from sklearn.model_selection import train_test_split
from torchlib.dataset.utils import create_data_loader


class ReplayBuffer(object):
    def __init__(self, capacity, obs_dtype, obs_shape, ac_dtype, ac_shape):
        self._capacity = capacity
        self._obs_dtype = obs_dtype
        self._obs_shape = obs_shape
        self._ac_dtype = ac_dtype
        self._ac_shape = ac_shape
        self._size = 0
        self._initialize()

    def __len__(self):
        return self._size

    def _initialize(self):
        """ Determine the inner data structure for storing """
        raise NotImplementedError

    def add(self, state, action, next_state, reward, done):
        """ For single environment """
        raise NotImplementedError

    def add_batch(self, states, actions, next_states, rewards, dones):
        """ For Vector Environment """
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def random_iterator(self, batch_size):
        raise NotImplementedError

    @property
    def state_mean_std(self):
        raise NotImplementedError

    @property
    def action_mean_std(self):
        raise NotImplementedError

    @property
    def delta_state_mean_std(self):
        raise NotImplementedError

    @property
    def reward_mean_std(self):
        raise NotImplementedError


class TransitionReplayBuffer(ReplayBuffer):
    """
    Store transitions in the replay buffer.
    The problem is
    1) If observation is image, then use trajectory-based replay for sampling
    2) Doesn't support windowed observations, use trajectory-based replay or wrap env with windowed obs.
    """

    def _initialize(self):
        # This will not explicitly allocate memory until access
        self._obs_storage = np.zeros(shape=[self._capacity] + list(self._obs_shape), dtype=self._obs_dtype)
        self._action_storage = np.zeros(shape=[self._capacity] + list(self._ac_shape), dtype=self._ac_dtype)
        self._next_obs_storage = np.zeros(shape=[self._capacity] + list(self._obs_shape), dtype=self._obs_dtype)
        self._reward_storage = np.zeros(shape=[self._capacity], dtype=np.float32)
        self._done_storage = np.zeros(shape=[self._capacity], dtype=np.bool)
        self._pointer = 0

    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._capacity
        self._size = min(self._size + count, self._capacity)

    def add(self, state, action, next_state, reward, done):
        self.add_batch(np.expand_dims(state, axis=0),
                       np.expand_dims(action, axis=0),
                       np.expand_dims(next_state, axis=0),
                       np.expand_dims(reward, axis=0),
                       np.expand_dims(done, axis=0))

    def add_batch(self, states, actions, next_states, rewards, dones):
        num_samples = states.shape[0]
        idx = np.arange(self._pointer, self._pointer + num_samples) % self._capacity
        self._obs_storage[idx] = states
        self._action_storage[idx] = actions
        self._next_obs_storage[idx] = next_states
        self._reward_storage[idx] = rewards
        self._done_storage[idx] = dones
        self._advance(num_samples)

    def sample(self, batch_size):
        idxes = np.random.choice(len(self), batch_size)
        return self._obs_storage[idxes], self._action_storage[idxes], self._next_obs_storage[idxes], \
               self._reward_storage[idxes], self._done_storage[idxes]

    def random_iterator(self, batch_size, train_val_split_ratio=0.2):
        """Create an iterator for the whole transitions
        Returns:
        """
        input_tuple = (self._obs_storage, self._action_storage, self._next_obs_storage,
                       self._reward_storage, self._done_storage)

        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)

        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]

        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=False)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=False,
                                             drop_last=False)

        return train_data_loader, val_data_loader

    @property
    def state_mean_std(self):
        obs = self._obs_storage[:len(self)]
        return np.mean(obs, axis=0), np.std(obs, axis=0)

    @property
    def action_mean_std(self):
        action = self._action_storage[:len(self)]
        return np.mean(action, axis=0), np.std(action, axis=0)

    @property
    def delta_state_mean_std(self):
        delta_state = self._next_obs_storage[:len(self)] - self._obs_storage[:len(self)]
        return np.mean(delta_state, axis=0), np.std(delta_state, axis=0)

    @property
    def reward_mean_std(self):
        reward = self._reward_storage[:len(self)]
        return np.mean(reward, axis=0), np.std(reward, axis=0)


Trajectory = namedtuple('Trajectory', ('state', 'action', 'reward', 'done'))


class FullTrajectoryReplayBuffer(ReplayBuffer):
    """
    Storing the full trajectory has the following advantages:
    1. Can create windowed states iterators
    2. Save memory for large state size (e.g. image)
    3. N-step bellman update
    4. Compute some intermediate results for PPO to update policy
    There are two ways to add data in trajectory replay buffer:
    1. Add full trajectory directly such as model-based approach and PPO.
    2. Add transition one by one such as DQN/DDPG/TD3 and SAC.
    """

    def _initialize(self):
        self.memory = deque()

    @property
    def num_trajectories(self):
        return len(self.memory)

    def add_trajectory(self, states, actions, rewards, done):
        self.memory.append(Trajectory(
            state=states,
            action=actions,
            reward=rewards.astype(np.float32),
            done=done
        ))
        self._size += actions.shape[0]

        while len(self) > self._capacity:
            trajectory = self.memory.popleft()
            self._size -= len(trajectory.action)

    def _create_state_action_next_state(self, window_length=None):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        if window_length is None:
            for trajectory in self.memory:
                states.append(trajectory.state[:-1])
                actions.append(trajectory.action)
                next_states.append(trajectory.state[1:])
                rewards.append(trajectory.reward)
                done = [False] * trajectory.action.shape[0]
                done[-1] = trajectory.done  # True done or not.
                dones.append(np.array(done))

            states = np.concatenate(states, axis=0)
            actions = np.concatenate(actions, axis=0)
            next_states = np.concatenate(next_states, axis=0)
            rewards = np.concatenate(rewards, axis=0)
            dones = np.concatenate(dones, axis=0)

        else:
            assert isinstance(window_length, int) and window_length > 1, \
                'window must be integer and greater than 2. Got {}'.format(window_length)
            for trajectory in self.memory:
                for i in range(window_length, trajectory.state.shape[0]):
                    states.append(trajectory.state[i - window_length:i])
                    next_states.append(trajectory.state[i])
                    actions.append(trajectory.action[i - window_length:i])
                rewards.append(trajectory.reward[window_length - 1:])
                done = [False] * (trajectory.action.shape[0] - window_length + 1)
                done[-1] = trajectory.done
                dones.append(np.array(done))

            states = np.stack(states, axis=0)
            actions = np.stack(actions, axis=0)
            next_states = np.stack(next_states, axis=0)
            rewards = np.concatenate(rewards, axis=0)
            dones = np.concatenate(dones, axis=0)

        return states, actions, next_states, rewards, dones

    def random_iterator(self, batch_size, train_val_split_ratio=0.2, window_length=None):
        """ Create iterator over (s, a, s', r, d)
        Args:
            batch_size: batch size
        Returns:
        """

        input_tuple = self._create_state_action_next_state(window_length=window_length)
        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)
        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]
        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=False)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=False,
                                             drop_last=False)
        return train_data_loader, val_data_loader

    def state_mean_std(self):
        states = []
        for trajectory in self.memory:
            states.append(trajectory.state)
        states = np.concatenate(states, axis=0)
        return np.mean(states, axis=0), np.std(states, axis=0)

    def action_mean_std(self):
        actions = []
        for trajectory in self.memory:
            actions.append(trajectory.action)
        actions = np.concatenate(actions, axis=0)
        return np.mean(actions, axis=0), np.std(actions, axis=0)

    def delta_state_mean_std(self):
        delta_states = []
        for trajectory in self.memory:
            states = trajectory.state
            delta_states.append(states[1:] - states[:-1])
        delta_states = np.concatenate(delta_states, axis=0)
        return np.mean(delta_states, axis=0), np.std(delta_states, axis=0)

    def reward_mean_std(self):
        rewards = []
        for trajectory in self.memory:
            rewards.append(trajectory.reward)
        rewards = np.concatenate(rewards, axis=0)
        return np.mean(rewards, axis=0), np.std(rewards, axis=0)


class StepTrajectoryReplayBuffer(ReplayBuffer):
    pass