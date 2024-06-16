import numpy as np
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle
import re
import torch
import torch.nn as nn
import torch.optim as optim
import random

# function to normalize payoffs in [0,1]

# function to normalize payoffs in [0,1]

def normalize_util(payoffs, min_payoff, max_payoff):
    if min_payoff == max_payoff:
        return payoffs
    payoff_range = max_payoff - min_payoff
    payoffs = np.maximum(payoffs, min_payoff)
    payoffs = np.minimum(payoffs, max_payoff)
    payoffs_scaled = (payoffs - min_payoff) / payoff_range
    return payoffs_scaled


normalize = np.vectorize(normalize_util)


# parent class of bidders
# parent class of bidders

class Bidder:
    def __init__(self, c_list, d_list, K, c_limit=None, d_limit=None, has_seed=False):
        self.K = K
        # if actions are provided
        if c_list and d_list:
            self.action_set = list(zip(c_list, d_list))
            self.cost = self.action_set[0]
        else:
            c_list = c_limit * np.random.sample(size=K-1)
            d_list = d_limit * np.random.sample(size=K-1)
            self.action_set = list(zip(c_list, d_list))
            # cost is a proper multiple of average bid function which is less than all of bid functions
            ratio_c = (c_list.min() / (2 * np.mean(c_list)))
            ratio_d = (d_list.min() / (2 * np.mean(d_list)))
            cost_ratio = min(ratio_c, ratio_d)
            self.cost = (np.mean(c_list) * cost_ratio, np.mean(d_list) * cost_ratio)
            
        self.weights = np.ones(K)
        self.history_payoff_profile = []
        self.history_action = []
        self.history_payoff = []
        self.cum_each_action = [0] * K
        self.played_action = None
        # to be able to reproduce exact same behavior
        self.has_seed = has_seed
        if self.has_seed:
            self.seed = np.random.randint(1, 10000)
            self.random_state = np.random.RandomState(seed=self.seed)

    # To clear stored data
    def restart(self):
        self.weights = np.ones(self.K)
        self.history_payoff_profile = []
        self.history_action = []
        self.history_payoff = []
        self.cum_each_action = [0] * self.K
        self.played_action = None
        if self.has_seed:
            self.random_state = np.random.RandomState(seed=self.seed)

    # choose action according to weights
    def choose_action(self):
        mixed_strategies = self.weights / np.sum(self.weights)
        if self.has_seed:
            choice = self.random_state.choice(len(self.action_set), p=mixed_strategies)
        else:
            choice = np.random.choice(len(self.action_set), p=mixed_strategies)
        return self.action_set[choice], choice

    # Player using Hedge algorithm (Freund and Schapire. 1997)

class Hedge_bidder(Bidder):
    def __init__(self, c_list, d_list, K, max_payoff, T, c_limit=None, d_limit=None, has_seed=False):
        super().__init__(c_list, d_list, K, c_limit=c_limit, d_limit=d_limit, has_seed=has_seed)
        self.type = 'Hedge'
        self.T = T
        self.learning_rate = np.sqrt(8 * np.log(self.K) / self.T)
        self.max_payoff = max_payoff

    def update_weights(self, payoffs):
        payoffs = normalize(payoffs, 0, self.max_payoff)
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.learning_rate, -losses)))
        self.weights = self.weights / np.sum(self.weights)
        
        # Player choosing actions uniformly random each time

class random_bidder(Bidder):
    def __init__(self, c_list, d_list, K, c_limit=None, d_limit=None, has_seed=False):
        super().__init__(c_list, d_list, K, c_limit=c_limit, d_limit=d_limit, has_seed=has_seed)
        self.type = 'random'
        
    def Update(self,payoffs):
        self.weights = self.weights 

class DQN_bidder(Bidder):
    def __init__(self, c_list, d_list, K, max_payoff, T, c_limit=None, d_limit=None, has_seed=False):
        super().__init__(c_list, d_list, K, c_limit=c_limit, d_limit=d_limit, has_seed=has_seed)
        self.type = 'dqn'
        self.T = T
        self.max_payoff = max_payoff
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.learning_rate = 5e-4
        self.action_size = K
        self.state_size = 2

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                act_values = self.model(state)
                return torch.argmax(act_values).item()

    def update_weights(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])
        done = torch.BoolTensor([done])

        if done:
            target = reward
        else:
            target = reward + self.gamma * torch.max(self.target_model(next_state))

        current = self.model(state)[action]
        loss = nn.functional.mse_loss(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()