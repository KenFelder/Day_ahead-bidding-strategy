import numpy as np
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
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

        # for same experience as DQN bidder
        self.min_replay_size = 0 # 200

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
        self.state_size = 3

        self.learning_rate = 5e-2
        self.gamma = 0 # 0.99
        self.batch_size = 100 # 500
        self.buffer_size = 50000
        self.min_replay_size = 0

        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 1
        self.target_update_freq = 5 # 10
        self.training_epochs = 1

        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.history_losses = []

        self.obs_scaler = MinMaxScaler()
        self.rews_scaler = MinMaxScaler()
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

        self.obs_scaler = MinMaxScaler()
        self.rews_scaler = MinMaxScaler()
        self.model, self.optimizer = self.build_model()  # Separate the model and the optimizer
        self.target_model, _ = self.build_model()  # Separate the model and the optimizer
        self.update_target_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            #nn.ReLU(),
            nn.Softmax(dim=1)
        )

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
#        print("Target model updated")

    def choose_action(self, epsilon, state):

        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            ind = np.random.choice(self.K)
            action = self.action_set[ind]
            return action, ind
        else:
            obs = np.asarray(state).reshape(1,-1)
            scaled_obs = self.obs_scaler.transform(obs)
            scaled_obs_t = torch.as_tensor(scaled_obs, dtype=torch.float32)
            q_values = self.model(scaled_obs_t)
            max_q_index = int(torch.argmax(q_values, dim=1)[0])
            action = self.action_set[max_q_index]
            self.weights = q_values.detach().numpy().reshape(-1)
            return action, max_q_index

    def update_weights(self):
        batch_size = min(self.batch_size, len(self.replay_buffer))
        transitions = random.sample(self.replay_buffer, batch_size)
        #transitions = random.sample(self.replay_buffer, len(self.replay_buffer))

        obses = np.asarray([t[0] for t in transitions])
        actions_ind = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions]).reshape(-1,1)
        new_obses = np.asarray([t[3] for t in transitions])

        # Scale the observations and rewards
        self.obs_scaler.fit(new_obses)
        scaled_obses = self.obs_scaler.transform(obses)
        scaled_new_obses = self.obs_scaler.transform(new_obses)
        self.rews_scaler.fit(rews)
        scaled_rews = self.rews_scaler.transform(rews)

        scaled_obses_t = torch.as_tensor(scaled_obses, dtype=torch.float32)
        actions_ind_t = torch.as_tensor(actions_ind, dtype=torch.int64).unsqueeze(-1)
        scaled_rews_t = torch.as_tensor(scaled_rews, dtype=torch.float32)
        scaled_new_obses_t = torch.as_tensor(scaled_new_obses, dtype=torch.float32)

        # Compute Targets
        target_q_values = self.target_model(scaled_new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = scaled_rews_t + self.gamma * max_target_q_values

        # Compute Loss
        q_values = self.model(scaled_obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_ind_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



        self.history_losses.append(loss.item())
        #self.weights = q_values.detach().numpy().reshape(-1)
        #print(self.weights)

#       print(f"loss: {loss}")

class DDPG_bidder(Bidder):
    def __init__(self, c_list, d_list, K, max_payoff, T, c_limit=None, d_limit=None, has_seed=False):
        super().__init__(c_list, d_list, K, c_limit=c_limit, d_limit=d_limit, has_seed=has_seed)
        self.type = 'ddpg'
        self.T = T
        self.max_payoff = max_payoff

        self.action_size = 2
        self.state_size = 3

        self.learning_rate = 5e-1
        self.gamma = 0  # 0.99
        self.batch_size = 15  # 500
        self.buffer_size = 50000
        self.min_replay_size = 0

        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 1
        self.target_update_freq = 5  # 10
        self.training_epochs = 50

        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.history_critic_losses = []
        self.history_actor_losses = []

        self.action_space_high_c = 1
        self.action_space_high_d = 20
        self.action_space_low_c = 0
        self.action_space_low_d = 0

        self.new_obs_scaler = MinMaxScaler()
        self.rews_scaler = MinMaxScaler()
        self.action_scaler = MinMaxScaler()
        self.action_scaler.fit(np.array(
            [[self.action_space_low_c, self.action_space_low_d], [self.action_space_high_c, self.action_space_high_d]]))

        self.actor_model, self.actor_optimizer = self.build_policy_model()
        self.target_actor_model, _ = self.build_policy_model()
        self.update_target_actor_model()

        self.critic_model, self.critic_optimizer = self.build_Q_model()  # Separate the model and the optimizer
        self.target_critic_model, _ = self.build_Q_model()  # Separate the model and the optimizer
        self.update_target_critic_model()

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

        self.new_obs_scaler = MinMaxScaler()
        self.rews_scaler = MinMaxScaler()
        self.action_scaler = MinMaxScaler()
        self.action_scaler.fit(np.array(
            [[self.action_space_low_c, self.action_space_low_d], [self.action_space_high_c, self.action_space_high_d]]))

        self.actor_model, self.actor_optimizer = self.build_policy_model()
        self.target_actor_model, _ = self.build_policy_model()  # Separate the model and the optimizer
        self.update_target_actor_model()

        self.critic_model, self.critic_optimizer = self.build_Q_model()  # Separate the model and the optimizer
        self.target_critic_model, _ = self.build_Q_model()  # Separate the model and the optimizer
        self.update_target_critic_model()

    def build_Q_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def build_policy_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Sigmoid()
        )

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def update_target_critic_model(self):
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())

    def update_target_actor_model(self):
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())

    def choose_action(self, epsilon, state):
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            c = random.uniform(self.action_space_low_c, self.action_space_high_c)
            d = random.uniform(self.action_space_low_d, self.action_space_high_d)
            action = (c, d)
            return action
        else:
            obs = np.asarray(state)
            scaled_obs = obs
            scaled_obs_t = torch.as_tensor(scaled_obs, dtype=torch.float32)
            print(f"actor input s_t: {scaled_obs}")
            action_t = self.actor_model(scaled_obs_t)
            action = action_t.detach().numpy()
            print(f"actor output pi(s_t) -> a_t: {action}")
            scaled_action_t = self.action_scaler.inverse_transform(action.reshape(1, -1))
            scaled_action = (scaled_action_t[0, 0], scaled_action_t[0, 1])
            print(f"scaled action: {scaled_action}")

            # Log
            action_np = action_t.detach().numpy()
            actor_input_np = scaled_obs_t.detach().numpy()
            print("Choose Action")
            print("#########################################################################################")
            print(f"actor input s_t: {actor_input_np}")
            print(f"actor output pi(s_t) -> a_t: {action_np}")
            print("#########################################################################################")


            return scaled_action

    def update_critic_weights(self):
        batch_size = min(self.batch_size, len(self.replay_buffer))
        transitions = random.sample(self.replay_buffer, batch_size)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions]).reshape(-1, 1)
        new_obses = np.asarray([t[3] for t in transitions])

        # Scale the actions, observations and rewards

        scaled_obses = obses
        self.new_obs_scaler.fit(np.asarray([t[3] for t in self.replay_buffer]))
        scaled_new_obses = self.new_obs_scaler.transform(new_obses)
        self.rews_scaler.fit(np.asarray([t[2] for t in self.replay_buffer]).reshape(-1, 1))
        scaled_rews = self.rews_scaler.transform(rews)
        scaled_actions = self.action_scaler.transform(actions)

        scaled_obses_t = torch.as_tensor(scaled_obses, dtype=torch.float32)
        scaled_actions_t = torch.as_tensor(scaled_actions, dtype=torch.float32)
        scaled_rews_t = torch.as_tensor(scaled_rews, dtype=torch.float32)
        scaled_new_obses_t = torch.as_tensor(scaled_new_obses, dtype=torch.float32)

        # Actor_Target(s_t+1) -> a_t+1
        next_state_actions_t = self.target_actor_model(scaled_new_obses_t)
        # Critic_Target(s_t+1, a_t+1) -> Q'(s_t+1, a_t+1)
        critic_next_target_t = self.target_critic_model(torch.cat((scaled_new_obses_t, next_state_actions_t), dim=1))
        # y = r + gamma * Q'(s_t+1, a_t+1)
        next_q_value_t = scaled_rews_t + self.gamma * critic_next_target_t

        # Critic(s_t, a_t) -> Q(s_t, a_t)
        q_a_values_t = self.critic_model(torch.cat((scaled_obses_t, scaled_actions_t), dim=1))
        # Loss = (Q(s_t, a_t) - y)^2
        q_loss = nn.functional.smooth_l1_loss(q_a_values_t, next_q_value_t)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        self.history_critic_losses.append(q_loss.item())

        # Log
        q_a_values_np = q_a_values_t.detach().numpy().reshape(-1)
        critic_scaled_input_np = torch.cat((scaled_obses_t, scaled_actions_t), dim=1).detach().numpy()
        critic_input_np = torch.cat((torch.as_tensor(obses), torch.as_tensor(actions)), dim=1).detach().numpy()
        print("Update Critic")
        for i in range(len(q_a_values_np)):
            print("#########################################################################################")
            print(f"critic input s_t, a_t: {critic_input_np[i]}")
            print(f"critic input scaled: {critic_scaled_input_np[i]}")
            print(f"q values: {q_a_values_np[i]}")
        print(f"critic loss: {q_loss}")
        print("#########################################################################################")

    def update_actor_weights(self):
        batch_size = min(self.batch_size, len(self.replay_buffer))
        transitions = random.sample(self.replay_buffer, batch_size)

        obses = np.asarray([t[0] for t in transitions])

        # Scale the observations
        scaled_obses = obses

        scaled_obses_t = torch.as_tensor(scaled_obses, dtype=torch.float32)

        actor_loss = -self.critic_model(torch.cat((scaled_obses_t, self.actor_model(scaled_obses_t)), dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.history_actor_losses.append(actor_loss.item())

        # Log
        actor_input_np = scaled_obses_t.detach().numpy()
        actor_output_np = self.actor_model(scaled_obses_t).detach().numpy()

        print("Update Actor")
        for i in range(len(actor_output_np)):
            print("#########################################################################################")
            print(f"actor input s_t: {actor_input_np[i]}")
            print(f"actor output pi(s_t) -> a_t: {actor_output_np[i]}")
            print("#########################################################################################")
        print(f"actor loss: {actor_loss}")