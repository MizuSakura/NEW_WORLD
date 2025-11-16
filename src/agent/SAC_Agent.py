import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.max_action = max_action

    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        def create_net():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.q1 = create_net()
        self.q2 = create_net()

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)
    

# ==============================
# 3. SAC Agent
# ==============================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.value_optim = optim.Adam(self.value.parameters(), lr=3e-4)

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state.reshape(1, -1))
        action, _ = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        s, a, r, s2, d = replay_buffer.sample(batch_size)
