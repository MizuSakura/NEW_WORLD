import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ==============================
# 1. Neural Network Definitions
# ==============================
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
        x_t = dist.rsample()  # reparameterization trick
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
# 2. Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s2),
            torch.FloatTensor(d).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ==============================
# 3. SAC Agent
# ==============================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.value = ValueNet(state_dim)
        self.target_value = ValueNet(state_dim)
        self.target_value.load_state_dict(self.value.state_dict())

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

        with torch.no_grad():
            next_v = self.target_value(s2)
            target_q = r + (1 - d) * self.gamma * next_v

        q1, q2 = self.critic(s, a)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        v = self.value(s)
        new_a, log_p = self.actor(s)
        q1_pi, q2_pi = self.critic(s, new_a)
        q_pi = torch.min(q1_pi, q2_pi)
        v_target = q_pi - self.alpha * log_p
        value_loss = ((v - v_target.detach()) ** 2).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        a_loss = (self.alpha * log_p - q_pi).mean()
        self.actor_optim.zero_grad()
        a_loss.backward()
        self.actor_optim.step()

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ==============================
# 4. Training & Plot
# ==============================
def train_sac(env_name="Pendulum-v1", episodes=150, max_steps=200):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    rewards = []
    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(replay_buffer) > 1000:
                agent.train(replay_buffer)

            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.2f}")

    env.close()

    # === Plot performance ===
    plt.plot(rewards)
    plt.title("SAC Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    # === Save model ===
    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    torch.save(agent.critic.state_dict(), "sac_critic.pth")
    print("✅ Model saved: sac_actor.pth, sac_critic.pth")

    return agent


# ==============================
# 5. Run Trained Model (Demo)
# ==============================
def evaluate(agent, env_name="Pendulum-v1", episodes=3):
    env = gym.make(env_name, render_mode="human")
    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
            if done:
                print(f"Eval Episode {ep+1}: Reward = {ep_reward:.2f}")
                break
    env.close()


if __name__ == "__main__":
    trained_agent = train_sac()
    evaluate(trained_agent)
