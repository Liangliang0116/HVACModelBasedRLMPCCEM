from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from torchlib.common import convert_numpy_to_tensor, move_tensor_to_gpu
from torchlib.dataset.utils import create_data_loader
from torchlib.deep_rl.algorithm.model_based import ModelBasedPlanAgent, ModelBasedDAggerAgent, ContinuousImitationPolicy


class ModelBasedHistoryPlanAgent(ModelBasedPlanAgent):
    def __init__(self, model, planner, window_length: int, baseline_agent):
        super(ModelBasedHistoryPlanAgent, self).__init__(model=model, planner=planner)
        self.history_states = deque(maxlen=window_length - 1)
        self.history_actions = deque(maxlen=window_length - 1)
        self.baseline_agent = baseline_agent

    def reset(self):
        """ Only reset on True done of one episode. """
        self.history_states.clear()

    def predict(self, state, weather_index):
        self.model.eval()
        if len(self.history_states) < self.history_states.maxlen:
            action = self.baseline_agent.predict(state)
        else:
            action = self.planner.predict(np.array(self.history_states), np.array(self.history_actions), state, weather_index)
        self.history_states.append(state)
        self.history_actions.append(action)
        self.model.train()
        return action


class StateActionPairDataset(object):
    def __init__(self, max_size):
        self.history_states = deque(maxlen=max_size)
        self.history_actions = deque(maxlen=max_size)
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)

    def __len__(self):
        return len(self.states)

    @property
    def maxlen(self):
        return self.states.maxlen

    @property
    def is_empty(self):
        return len(self) == 0

    def add(self, history_state, history_action, state, action):
        self.history_states.append(history_state)
        self.history_actions.append(history_action)
        self.states.append(state)
        self.actions.append(action)

    def random_iterator(self, batch_size, train_val_split_ratio=0.2):
        history_states = np.array(self.history_states)
        history_actions = np.array(self.history_actions)
        states = np.array(self.states)
        actions = np.array(self.actions)

        input_tuple = (history_states, history_actions, states, actions)

        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)

        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]

        # in training, we drop last batch to avoid batch size 1 that may crash batch_norm layer.
        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=False,
                                             drop_last=False)

        return train_data_loader, val_data_loader


class LSTMImitationModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(LSTMImitationModule, self).__init__()
        self.discrete = False
        feature_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=1, batch_first=True,
                            bidirectional=False)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size + state_dim, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, history_states, history_actions, states):
        feature = torch.cat((history_states, history_actions), dim=-1)
        output, _ = self.lstm.forward(feature)
        output = self.drop_out.forward(output)
        output = output[:, -1, :]
        output = torch.cat((output, states), dim=-1)
        output = self.linear.forward(output)
        return output


class HistoryImitationPolicy(ContinuousImitationPolicy):
    def __init__(self, state_dim, action_dim, hidden_size=32, learning_rate=1e-3):
        model = LSTMImitationModule(state_dim, action_dim, hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super(HistoryImitationPolicy, self).__init__(model=model, optimizer=optimizer)

    def predict(self, history_state, history_action, state):       
        state = np.expand_dims(state, axis=0)
        history_state = np.expand_dims(history_state, axis=0)
        history_action = np.expand_dims(history_action, axis=0)
        with torch.no_grad():
            state = convert_numpy_to_tensor(state)
            history_state = convert_numpy_to_tensor(history_state)
            history_action = convert_numpy_to_tensor(history_action)
            state = (state - self.state_mean.squeeze(dim=1)) / self.state_std.squeeze(dim=1)
            history_state = (history_state - self.state_mean) / self.state_std
            action = self.model.forward(history_state, history_action, state)
        return action.cpu().numpy()[0]

    def fit(self, dataset: StateActionPairDataset, epoch=10, batch_size=128, verbose=False):
        t = range(epoch)
        if verbose:
            t = tqdm(t)

        train_data_loader, val_data_loader = dataset.random_iterator(batch_size=batch_size)

        for i in t:
            losses = []
            for history_states, history_actions, states, actions in train_data_loader:
                self.optimizer.zero_grad()
                history_states = move_tensor_to_gpu(history_states)
                history_actions = move_tensor_to_gpu(history_actions)
                states = move_tensor_to_gpu(states)
                actions = move_tensor_to_gpu(actions)

                history_states = (history_states - self.state_mean) / self.state_std
                states = (states - self.state_mean.squeeze(dim=1)) / self.state_std.squeeze(dim=1)

                output = self.model.forward(history_states, history_actions, states)
                loss = self.loss_fn(output, actions)  # This is for imitation learning, so the loss function is about actions. 
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                for history_states, history_actions, states, actions in val_data_loader:
                    history_states = move_tensor_to_gpu(history_states)
                    history_actions = move_tensor_to_gpu(history_actions)
                    states = move_tensor_to_gpu(states)
                    actions = move_tensor_to_gpu(actions)
                    history_states = (history_states - self.state_mean) / self.state_std
                    states = (states - self.state_mean.squeeze(dim=1)) / self.state_std.squeeze(dim=1)
                    output = self.model.forward(history_states, history_actions, states)
                    loss = self.loss_fn(output, actions)
                    val_losses.append(loss.item())

            self.train()

            if verbose:
                t.set_description('Epoch {}/{} - Avg policy train loss: {:.4f} - Avg policy val loss: {:.4f}'.format(
                    i + 1, epoch, np.mean(losses), np.mean(val_losses)))


class ModelBasedHistoryDaggerAgent(ModelBasedDAggerAgent):
    def __init__(self, model, planner, policy_data_size, window_length, baseline_agent, 
                 state_dim, action_dim, hidden_size=32):
        """ Initialize an instance of the class.

        Args:
            model: learned dynamics model
            planner: MPC planner to obtain an action under a certain state and historical observations
            policy_data_size (int): maximum of the dataset to train the network of imitation learning 
            window_length (int): length of historical data
            baseline_agent: agent to collect initial dataset for imitation learning
            state_dim (int): state dimension of the environment
            action_dim (int): action dimension of the environment
            hidden_size (int, optional): hidden size of the neural network of imitation learning. Defaults to 32.
        """
        policy = HistoryImitationPolicy(state_dim=state_dim, 
                                        action_dim=action_dim, 
                                        hidden_size=hidden_size)
        super(ModelBasedHistoryDaggerAgent, self).__init__(model, planner, policy, policy_data_size)
        self.history_states = deque(maxlen=window_length - 1)
        self.history_actions = deque(maxlen=window_length - 1)
        self.baseline_agent = baseline_agent
        self.state_action_dataset = StateActionPairDataset(max_size=policy_data_size)

    def predict(self, state, weather_index):
        """[summary]

        Args:
            state (np.ndarray): state for which an action needs to be obtained
            weather_index (int): weather index

        Returns:
            np.ndarray: action to be executed
        """
        if len(self.history_states) < self.history_states.maxlen:
            action = self.baseline_agent.predict(state)
        else:
            history_state = np.array(self.history_states)
            history_action = np.array(self.history_actions)
            action = self.planner.predict(history_state, history_action, state, weather_index)
            self.state_action_dataset.add(history_state=history_state,
                                          history_action=history_action,
                                          state=state,
                                          action=action)
            self.policy.eval()
            action = self.policy.predict(history_state, history_action, state)
            self.policy.train()
        
        self.history_states.append(state)
        self.history_actions.append(action)
        return action

    def fit_policy(self, epoch=10, batch_size=128, verbose=False):
        """ Fit the policy

        Args:
            epoch (int, optional): training epochs for policy. Defaults to 10.
            batch_size (int, optional): batch size for training policy. Defaults to 128.
            verbose (bool, optional): [description]. Defaults to False.
        """
        if len(self.state_action_dataset) > 0:
            self.policy.train()
            self.policy.fit(dataset=self.state_action_dataset, 
                            epoch=epoch, 
                            batch_size=batch_size,
                            verbose=verbose)
