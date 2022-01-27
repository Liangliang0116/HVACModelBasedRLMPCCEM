import torch
import torch.nn as nn
import os

from .models import RLNN
    

class Actor(RLNN):
    def __init__(self, state_dim, action_dim, max_action, mpc_horizon, layer_norm, actor_lr):
        super(Actor, self).__init__(state_dim, action_dim, max_action)
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
        self.dropout2 = nn.Dropout(0.25)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.ReLU()
        self.linear = nn.Sequential(
            nn.Linear(hidden_size_lstm+out_features_f2, hidden_size_linear),  # NOTE: hidden_size_linear is used twice here. 
            nn.ReLU6(),
            nn.Linear(hidden_size_linear, action_dim),
            nn.Tanh()
        )

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=actor_lr)
    
    def forward(self, history_states, history_actions, current_state, weather_pred):
        feature = torch.cat((history_states, history_actions), dim=-1) 
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
        current_state, history_obses, _, history_actions, weather_pred, _ = memory.sample(batch_size)
        actor_loss = -critic(current_state=current_state, 
                             history_states=history_obses,
                             current_action=self(history_obses, history_actions, current_state, weather_pred),
                             history_actions=history_actions,
                             weather_pred=weather_pred
                             ).mean() 
        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer.step()


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, mpc_horizon, layer_norm, critic_lr):
        super(Critic, self).__init__(state_dim, action_dim, 1)
        feature_dim = state_dim + action_dim
        out_features_f1 = 128
        hidden_size_lstm = 128
        hidden_size_linear = 128
        out_features_f2 = 128
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=out_features_f1)
        self.lstm = nn.LSTM(input_size=out_features_f1, hidden_size=hidden_size_lstm, num_layers=1, 
                            batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.25)
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
        
    def forward(self, current_state, history_states, current_action, history_actions, weather_pred):
        feature = torch.cat((history_states, history_actions), dim=-1) 
        output = self.fc1(feature)
        # output = self.relu1(output)
        output, _ = self.lstm(output)
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
        current_state, history_obses, current_action, history_actions, weather_pred, score = memory.sample(batch_size)
        target_Q = score
        current_Q = self(current_state, history_obses, current_action, history_actions, weather_pred)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optimizer.step()
