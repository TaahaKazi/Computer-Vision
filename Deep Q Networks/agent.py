import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action
            a = torch.Tensor(random.sample(range(self.action_size), 1)).to(torch.int)  # since a will be used as an index, we need to covnert the tensor's dtype to int
        else:
            ### CODE ####
            # Choose the best action
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            a = self.policy_net(state_tensor).max(1)[1]
            a = a.cpu()
        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)


        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        #print("Q:value:", self.policy_net(states))
        #print("Q:value-max(1)[0]:",self.policy_net(states).max(1)[0])
        #print("q val shape:", self.policy_net(states).shape)
        #print("actions shape:", actions.shape)
        Q_value = self.policy_net(states).gather(1, actions.unsqueeze(1))  # map the actions taken (which includes stochaticity) onto the corresponding the Q-value

        # Compute Q function of next state
        ### CODE ####
        with torch.no_grad():  # so that gradients of Q_func_next wrt weights of policy_net aren't computed & accumulated
          next_states = torch.from_numpy(next_states).cuda()
          Q_func_next = self.policy_net(next_states)

        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        Q_value_next = Q_func_next.max(1)[0]
        mask = mask.to(device)
        Q_value_next = Q_value_next * mask  # all terminal next_states should have 0 Q-Value

        # Compute the Huber Loss
        ### CODE ####
        Q_value_target = rewards + self.discount_factor * Q_value_next
        criterion = nn.HuberLoss()
        loss = criterion(Q_value, Q_value_target)

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

