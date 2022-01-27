import numpy as np
import torch
import torch.multiprocessing as mp

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

# Code based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/jingweiz/pytorch-distributed/blob/master/core/memories/shared_memory.py


class Memory():

    def __init__(self, memory_size, state_dim, action_dim, window_length, mpc_horizon):

        # params
        self.memory_size = memory_size      
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_length = window_length
        self.mpc_horizon = mpc_horizon
        self.pos = 0
        self.full = False

        if USE_CUDA:
            self.current_states = torch.zeros(self.memory_size, self.state_dim).cuda()
            self.history_states = torch.zeros(self.memory_size, self.window_length, self.state_dim).cuda()
            self.current_actions = torch.zeros(self.memory_size, self.action_dim).cuda()
            self.history_actions = torch.zeros(self.memory_size, self.window_length, self.action_dim).cuda()
            self.weather_preds = torch.zeros(self.memory_size, self.mpc_horizon).cuda()
            self.scores = torch.zeros(self.memory_size, 1).cuda()
        else:
            self.current_states = torch.zeros(self.memory_size, self.state_dim)
            self.history_states = torch.zeros(self.memory_size, self.window_length, self.state_dim)
            self.current_actions = torch.zeros(self.memory_size, self.action_dim)
            self.history_actions = torch.zeros(self.memory_size, self.window_length, self.action_dim)
            self.weather_preds = torch.zeros(self.memory_size, self.mpc_horizon)
            self.scores = torch.zeros(self.memory_size, 1)
            

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def get_pos(self):
        return self.pos

    # Expects tuples of (state, next_state, action, reward, done)

    def add(self, datum):

        current_state, history_states, current_action, history_actions, weather_preds, scores = datum
        # print('type(current_state) =', type(current_state))
        # print('type(history_states) =', type(history_states))
        # print('type(current_action) =', type(current_action))
        # print('type(history_actions) =', type(history_actions))
        # print('type(weather_preds) =', type(weather_preds))
        # print('type(scores) =', type(scores))
        # raise ValueError()

        self.current_states[self.pos] = FloatTensor(current_state)
        self.history_states[self.pos] = FloatTensor(history_states)
        self.current_actions[self.pos] = FloatTensor(current_action)
        self.history_actions[self.pos] = FloatTensor(history_actions)
        self.weather_preds[self.pos] = FloatTensor(weather_preds)
        self.scores[self.pos] = FloatTensor([scores])

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        upper_bound = self.memory_size if self.full else self.pos     
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.current_states[batch_inds],
                self.history_states[batch_inds],
                self.current_actions[batch_inds],
                self.history_actions[batch_inds],
                self.weather_preds[batch_inds],
                self.scores[batch_inds])

    def get_reward(self, start_pos, end_pos):

        tmp = 0
        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):
                tmp += self.rewards[i]
        else:
            for i in range(start_pos, self.memory_size):
                tmp += self.rewards[i]

            for i in range(end_pos):
                tmp += self.rewards[i]

        return tmp

    def repeat(self, start_pos, end_pos):

        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):
                
                self.current_states[self.pos] = self.current_states[i].clone()
                self.history_states[self.pos] = self.history_states[i].clone()
                self.current_actions[self.pos] = self.current_actions[i].clone()
                self.history_actions[self.pos] = self.history_actions[i].clone()
                self.weather_preds[self.pos] = self.weather_preds[i].clone()
                self.scores[self.pos] = self.scores[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0

        else:
            for i in range(start_pos, self.memory_size):

                self.current_states[self.pos] = self.current_states[i].clone()
                self.history_states[self.pos] = self.history_states[i].clone()
                self.current_actions[self.pos] = self.current_actions[i].clone()
                self.history_actions[self.pos] = self.history_actions[i].clone()
                self.weather_preds[self.pos] = self.weather_preds[i].clone()
                self.scores[self.pos] = self.scores[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0

            for i in range(end_pos):

                self.current_states[self.pos] = self.current_states[i].clone()
                self.history_states[self.pos] = self.history_states[i].clone()
                self.current_actions[self.pos] = self.current_actions[i].clone()
                self.history_actions[self.pos] = self.history_actions[i].clone()
                self.weather_preds[self.pos] = self.weather_preds[i].clone()
                self.scores[self.pos] = self.scores[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0


class SharedMemory():

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = mp.Value('l', 0)
        self.full = mp.Value('b', False)

        self.states = FloatTensor(torch.zeros(
            self.memory_size, self.state_dim))
        self.actions = FloatTensor(torch.zeros(
            self.memory_size, self.action_dim))
        self.n_states = FloatTensor(
            torch.zeros(self.memory_size, self.state_dim))
        self.rewards = FloatTensor(torch.zeros(self.memory_size, 1))
        self.dones = FloatTensor(torch.zeros(self.memory_size, 1))

        self.states.share_memory_()
        self.actions.share_memory_()
        self.n_states.share_memory_()
        self.rewards.share_memory_()
        self.dones.share_memory_()

        self.memory_lock = mp.Lock()

    def size(self):
        if self.full.value:
            return self.memory_size
        return self.pos.value

    # Expects tuples of (state, next_state, action, reward, done)

    def _add(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos.value] = FloatTensor(state)
        self.n_states[self.pos.value] = FloatTensor(n_state)
        self.actions[self.pos.value] = FloatTensor(action)
        self.rewards[self.pos.value] = FloatTensor([reward])
        self.dones[self.pos.value] = FloatTensor([done])

        self.pos.value += 1
        if self.pos.value == self.memory_size:
            self.full.value = True
            self.pos.value = 0

    def add(self, experience):
        with self.memory_lock:
            return self._add(experience)

    def _sample(self, batch_size):

        upper_bound = self.memory_size if self.full.value else self.pos.value
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])

    def sample(self, batch_size):
        with self.memory_lock:
            return self._sample(batch_size)

    def repeat(self, start_pos, end_pos):

        for i in range(start_pos, end_pos):

            self.states[self.pos.value] = self.states[i].clone()
            self.n_states[self.pos.value] = self.n_states[i].clone()
            self.actions[self.pos.value] = self.actions[i].clone()
            self.rewards[self.pos.value] = self.rewards[i].clone()
            self.dones[self.pos.value] = self.dones[i].clone()

            self.pos.value += 1
            if self.pos.value == self.memory_size:
                self.full.value = True
                self.pos.value = 0

        print(self.states.size())


class Archive(list):

    def __init__(self):

        # counter
        self.cpt = 0

    def add_sample(self, sample):

        # adding the sample to the archive
        self.append(sample)
        self.cpt += 1

    def add_samples(self, samples):

        # adding the samples to the archive
        for sample in samples:
            self.add_sample(sample)

    def add_gen(self, idx, gen):

        self[idx].gens.append(gen)

    def get_size(self):
        return min(self.max_size, self.cpt)
