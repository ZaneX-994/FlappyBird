import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.9  # discount rate
        self.epsilon = 0.1 # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.0001
        self.learning_rate = 0.001
        self.learning_rate_decay = 0.995
        self.learning_rate_min = 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.__build_model().to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def __build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 4),
            nn.LeakyReLU(),
            nn.Linear(4, self.action_size),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.model(states_tensor).gather(1, actions_tensor)
        next_q_values = self.model(next_states_tensor).max(1)[0].unsqueeze(1)
        target_q_values = rewards_tensor + self.gamma * next_q_values * (~dones_tensor)

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay

    def save_state(self):
        torch.save(self.model.state_dict(), 'trained_q_agent.pth')

    def load_state(self):
        self.model.load_state_dict(torch.load('trained_q_agent.pth'))
