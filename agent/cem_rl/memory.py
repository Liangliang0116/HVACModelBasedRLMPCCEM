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


class Memory:

    def __init__(self, memory_size, state_dim, action_dim, window_length, mpc_horizon):
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

    def add(self, datum):
        current_state, history_states, current_action, history_actions, weather_preds, scores = datum
        
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
