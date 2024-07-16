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
from collections import deque


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

class Bidder:
    def __init__(self, c_list, d_list, K, c_limit=None, d_limit=None, has_seed=False):
        self.K = K
        # if actions are provided
        if c_list and d_list:
            self.action_set = list(zip(c_list, d_list))
            self.cost = self.action_set[0]
        else:
            c_list = c_limit * np.random.sample(size=K - 1)
            d_list = d_limit * np.random.sample(size=K - 1)
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

    def Update(self, payoffs):
        self.weights = self.weights


class DQN_bidder(Bidder):
    def __init__(self, c_list, d_list, K, max_payoff, T, c_limit=None, d_limit=None, has_seed=False):
        super().__init__(c_list, d_list, K, c_limit=c_limit, d_limit=d_limit, has_seed=has_seed)
        self.type = 'dqn'
        self.T = T
        self.max_payoff = max_payoff

        self.action_size = K
        self.state_size = 2

        self.learning_rate = 5e-2
        self.gamma = 0.99
        self.batch_size = 500
        self.buffer_size = 50000
        self.min_replay_size = 750

        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 20
        self.target_update_freq = 10

        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.history_losses = []

        self.model, self.optimizer = self.build_model()  # Separate the model and the optimizer
        self.target_model, _ = self.build_model()  # Separate the model and the optimizer
        self.update_target_model()

    def restart(self):
        self.weights = np.ones(self.K)
        self.history_payoff_profile = []
        self.history_action = []
        self.history_payoff = []
        self.cum_each_action = [0] * self.K
        self.played_action = None
        if self.has_seed:
            self.random_state = np.random.RandomState(seed=self.seed)

        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.model, self.optimizer = self.build_model()  # Separate the model and the optimizer
        self.target_model, _ = self.build_model()  # Separate the model and the optimizer
        self.update_target_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.Linear(64, 64),
            nn.Linear(64, self.action_size)
            #nn.Softmax(1)
        )

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
#        print("Target model updated")

    def choose_action(self, epsilon, state):

        rnd_sample = random.random()
        #print(f"rnd_sample: {rnd_sample}, epsilon: {epsilon}")
        if rnd_sample <= epsilon:
            ind = np.random.choice(self.K)
            action = self.action_set[ind]
            return action, ind
        else:
            obs_t = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(obs_t.unsqueeze(0))
            max_q_index = int(torch.argmax(q_values, dim=1)[0])
            action = self.action_set[max_q_index]
            print(f"action: {action}, argmax: {max_q_index, q_values}")
            return action, max_q_index

    def update_weights(self):
        transitions = random.sample(self.replay_buffer, self.batch_size)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = np.asarray([t[3] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        target_q_values = self.target_model(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + self.gamma * max_target_q_values

        # Compute Loss
        q_values = self.model(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.history_losses.append(loss.item())

        print(f"loss: {loss}")
