import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from aux_functions import Bidder,  random_bidder, Hedge_bidder, DQN_bidder, DDPG_bidder
from tqdm import tqdm
import pickle
import re
import csv


class auction_data:
    def __init__(self):
        self.bids = []
        self.bids_ind = []
        self.allocations = []
        self.payments = []
        self.marginal_prices = []
        self.payoffs = []
        self.regrets = []
        self.Q = []
        self.SW = []
        self.final_dist = None
        self.losses = []
        
        
        
        # estimates maximum payoff from results of a random play
def calc_max_payoff(Q, c_list, d_list, N, T, K, cap):
    num_games = 10
    num_runs = 10
    game_data_profile = []
    for i in range(num_games):
        bidders = []
        for i in range(N):
            bidders.append(random_bidder(c_list[i], d_list[i], K))
        for run in range(num_runs):
            game_data_profile.append(run_auction(T, bidders, Q, cap, regret_calc=False).payoffs)
    return np.max(np.array(game_data_profile))

# simulates the selection process in the auction

def import_load(file_name):
    Q_forecast = []
    Q_load = []
    with open(file_name, newline='') as file:
        data = csv.reader(file, delimiter=',', quotechar='"')
        for row in data:
            Q_forecast.append(row[0].replace('"', '').split(',')[1])
            Q_load.append(row[0].replace('"', '').split(',')[2])
    Q_load = [int(row) for row in Q_load[1:]]
    Q_forecast = [int(row) for row in Q_forecast[1:]]
    return Q_forecast, Q_load

def optimize_alloc(bids, Q, cap):
    C = np.array([param[0] for param in bids])
    C = np.diag(C)
    D = np.array([param[1] for param in bids])
    n = len(bids)
    A = np.ones(n).T
    G = - np.eye(n)
    h = np.zeros(n)
    I = np.eye(n)

    # non-negativity doesn't strictly hold (small negative allocations might occur)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, C) + D.T @ x),
                      [G @ x <= h, A @ x == Q, I @ x <= cap])
    prob.solve()
    allocs = x.value
    social_welfare = prob.value
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 10 ** (-5):
            allocs[i] = 0

    # only for quadratic case
    sample_winner = np.argmin(allocs)
    marginal_price = bids[sample_winner][0] * min(allocs) + bids[sample_winner][1]
    payments = marginal_price * allocs

    return allocs, marginal_price, payments, social_welfare

# runs a repeated auction

def optimize_welfare(bids, cap, a=1, b=1500):
    # Define problem parameters
    n = len(bids)  # Number of variables

    # Define the optimization variable
    x = cp.Variable(n)

    c = np.array([param[0] for param in bids])
    C = np.diag(c)
    d = np.array([param[1] for param in bids])
    I = np.eye(n)

    # Define the objective function
    objective = cp.Maximize(
        -0.5 * a * cp.square(cp.sum(x)) + b * cp.sum(x) - (0.5 * cp.quad_form(x, C) + d.T @ x)
    )

    # Define the constraints
    constraints = [I @ x <= cap, x >= 0]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()
    allocs = x.value
    social_welfare = problem.value
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 10 ** (-1):
            allocs[i] = 0
    prices = C @ x + d
    sample_winner = np.argmax([prices.value[i] if allocs[i] > 0 else 0 for i in range(n)])
    marginal_price = np.max(prices.value[sample_winner])
    payments = marginal_price * allocs

    return allocs, marginal_price, payments, social_welfare

def run_auction(T, bidders, Q, cap, regret_calc, regret_all=False):
    for b in bidders:
        b.restart()
    
    game_data = auction_data()
#     player_final_dists = []
    for t in range(T):
        bids = []
        for bidder in bidders:
            action, ind = bidder.choose_action()
            bidder.played_action = action
            bidder.history_action.append(ind)
            bids.append(action)

        x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)

        # calculates payoffs from payments
        payoff = []
        for i, bidder in enumerate(bidders):
            payoff_bidder = payments[i] - (0.5 * bidder.cost[0] * x[i] + bidder.cost[1]) * x[i]
            payoff.append(payoff_bidder)
            bidder.history_payoff.append(payoff_bidder)
        game_data.payoffs.append(payoff)

        # calculates real regret for all bidders/ Hedge also needs this part for its update
        if regret_calc:
            regrets = []
            for i, bidder in enumerate(bidders):
                if regret_all or (not regret_all and i == len(bidders) - 1):
                    payoffs_each_action = []
                    for j, action in enumerate(bidder.action_set):
                        tmp_bids = bids.copy()
                        tmp_bids[i] = action
                        x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                        payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                        payoffs_each_action.append(payoff_action)
                        bidder.cum_each_action[j] += payoff_action
                    bidder.history_payoff_profile.append(np.array(payoffs_each_action))
                    regrets.append(
                        (max(bidder.cum_each_action) - sum(bidder.history_payoff))/(t+1))
#                     bidder.update_weights(bidder.history_payoff_profile[-1])

            # update weights
            for i, bidder in enumerate(bidders):
                if bidder.type == 'Hedge':
                    bidder.update_weights(bidder.history_payoff_profile[t])
                if bidder.type == 'EXP3':
                    bidder.update_weights(bidder.history_action[t], bidder.history_payoff[t])
                if bidder.type == 'GPMW':
                    bidder.update_weights(x[i], marginal_price)
                if bidder.type == 'DQN':
                    bidder.update_weights()
            game_data.regrets.append(regrets)

        # store data
        game_data.Q.append(Q)
        game_data.SW.append(social_welfare)
        game_data.bids.append(bids)
        game_data.allocations.append(x)
        game_data.payments.append(payments)
        game_data.payoffs.append(payoff)
        game_data.marginal_prices.append(marginal_price)
    game_data.final_dist = bidders[len(bidders) - 1].weights
    
    return game_data
    
def combine(res, res_list):
    game_data_profile = []
    for name in res_list:
        with open(f'{name}.pckl', 'rb') as file:
            T = pickle.load(file)
            types = pickle.load(file)
            game_data_profile.append(pickle.load(file))
    
    total_len = len(game_data_profile)    
    combined_profile = []
    for i in range(len(types)):
        tmp = []
        for r in range(total_len):
            tmp += game_data_profile[r][i]
        combined_profile.append(tmp)
    
    with open(f'{res}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(combined_profile, file)
        
        
    # Case a ==> Trustful Vs Hedge (Others play Trustful vs B5 Plays Hedge)  


# Case 0 ==> Trustful Vs DQN

def Trustful_vs_DDPG(num_games, num_runs, T, file_name, seed=20, static_profiles=False, welfare=True):
    types = ['Trustful vs DDPG']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_cost_DDPG = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]  # last player
    d_cost_DDPG = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]

    # Actions of others obtained from diagonalization + their true cost
    DQN_profile = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000

    ddpg_bidder = DDPG_bidder(c_cost_DDPG, d_cost_DDPG, K, max_payoff, T)

    player_final_dists = []
    for run in tqdm(range(num_runs)):

        # initialize
        ddpg_bidder.restart()
        game_data = auction_data()

        # Training Loop / Game
        x = [0, 0, 0, 0, 0]
        marginal_price = 0
        Q = Q
        for t in range(T):
            if static_profiles:
                Q = Q
                cap = cap
            else:
                cap = 700
                cap = [np.random.normal(loc=cap, scale=0.15*cap, size=T) for _ in range(5)]
                cap = [cap[i][t] for i in range(5)]
                Q = 1448.4
                np.random.seed(seed)
                Q_load = np.random.normal(loc=Q, scale=0.15*Q, size=T)
                Q = Q_load[t]
            epsilon = np.interp(t, [0, ddpg_bidder.epsilon_decay], [ddpg_bidder.epsilon_start, ddpg_bidder.epsilon_end])
            action = ddpg_bidder.choose_action(epsilon, [0, 0, 0])
            ddpg_bidder.played_action = action
            ddpg_bidder.history_action.append(action)
            bids = DQN_profile + [action]
            #print(bids)
            if welfare:
                x, marginal_price, payments, social_welfare = optimize_welfare(bids, cap)
            else:
                x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)
            payoff = []
            for i in range(N):
                if i == N - 1:
                    payoff_DDPG = payments[-1] - (0.5 * ddpg_bidder.cost[0] * x[-1] + ddpg_bidder.cost[1]) * x[-1]
                    ddpg_bidder.history_payoff.append(payoff_DDPG)
                    payoff.append(payoff_DDPG)
                    #print(f'Run {run}) DQN player payoff at round {t + 1}: {payoff_DQN}')
                #                     if t==T-1:
                #                         print(f'Run {run}) DQN player payoff at round {t+1}: {payoff_DQN}')
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)

            state = [0, 0, 0]
            #state = budget
            action = action
            reward = payoff_DDPG
            #state_new = [x[-1], marginal_price, Q]
            state_new = [x[-1], marginal_price, Q]

            transition = (state, action, reward, state_new)

            ddpg_bidder.replay_buffer.append( transition )



            bidder = ddpg_bidder
            i = -1
            payoffs_each_action = []
            for ind, action in enumerate(bidder.action_set):
                tmp_bids = bids.copy()
                tmp_bids[i] = action
                if welfare:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_welfare(tmp_bids, cap)
                else:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                payoffs_each_action.append(payoff_action)
                bidder.cum_each_action[ind] += payoff_action
                state_new = [x_tmp[-1], marginal_price_tmp, Q]
                bidder.replay_buffer.append((state, action, payoff_action, state_new))
                #bidder.replay_buffer.append((state, action, payoff_action, state + payoff_action))
            bidder.history_payoff_profile.append(np.array(payoffs_each_action))
            regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff)) / (t + 1)
            for i in range(bidder.training_epochs):
                bidder.update_critic_weights()

            if t % bidder.target_update_freq == 0:
                for i in range(bidder.training_epochs):
                    bidder.update_actor_weights()
                bidder.update_target_actor_model()
                bidder.update_target_critic_model()

            budget = state_new

            x_old = x
            marginal_price_old = marginal_price
            Q_old = Q

            game_data.regrets.append([regret])

            # store data
            game_data.Q.append(Q)
            game_data.SW.append(social_welfare)
            game_data.bids.append(bids)
            game_data.allocations.append(x)
            game_data.payments.append(payments)
            game_data.marginal_prices.append(marginal_price)

        player_final_dists.append(ddpg_bidder.weights)
        game_data_profile[0].append(game_data)

    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
        pickle.dump(player_final_dists, file)

def Trustful_vs_DQN(num_games, num_runs, T, file_name, seed=20, static_profiles=False, welfare=True):
    types = ['Trustful vs DQN']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_cost_DQN = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]  # last player
    d_cost_DQN = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]

    # Actions of others obtained from diagonalization + their true cost
    DQN_profile = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000

    dqn_bidder = DQN_bidder(c_cost_DQN, d_cost_DQN, K, max_payoff, T)

    player_final_dists = []
    for run in tqdm(range(num_runs)):

        # reset game
        dqn_bidder.restart()
        game_data = auction_data()

        # Training Loop / Game
        x = [0, 0, 0, 0, 0]
        marginal_price = 0
        for t in range(T):
            if static_profiles:
                Q = Q
                cap = cap
            else:
                cap = 700
                cap = [np.random.normal(loc=cap, scale=0.15*cap, size=T) for _ in range(5)]
                cap = [cap[i][t] for i in range(5)]
                Q = 1448.4
                np.random.seed(seed)
                Q_load = np.random.normal(loc=Q, scale=0.15*Q, size=T)
                Q = Q_load[t]
            epsilon = np.interp(t, [0, dqn_bidder.epsilon_decay], [dqn_bidder.epsilon_start, dqn_bidder.epsilon_end])
            action, ind = dqn_bidder.choose_action(epsilon, [x[-1], marginal_price, Q])
            dqn_bidder.played_action = action
            dqn_bidder.history_action.append(ind)
            bids = DQN_profile + [action]
            inds = [0, 0, 0, 0, ind]
            #print(bids)
            if welfare:
                x, marginal_price, payments, social_welfare = optimize_welfare(bids, cap)
            else:
                x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)


            payoff = []
            for i in range(N):
                if i == N - 1:
                    payoff_DQN = payments[-1] - (0.5 * dqn_bidder.cost[0] * x[-1] + dqn_bidder.cost[1]) * x[-1]
                    dqn_bidder.history_payoff.append(payoff_DQN)
                    payoff.append(payoff_DQN)
                    #print(f'Run {run}) DQN player payoff at round {t + 1}: {payoff_DQN}')
                #                     if t==T-1:
                #                         print(f'Run {run}) DQN player payoff at round {t+1}: {payoff_DQN}')
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)



            bidder = dqn_bidder
            i = -1
            payoffs_each_action = []
            for ind, action in enumerate(bidder.action_set):
                tmp_bids = bids.copy()
                tmp_bids[i] = action
                if welfare:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_welfare(tmp_bids, cap)
                else:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                payoffs_each_action.append(payoff_action)
                bidder.cum_each_action[ind] += payoff_action
                dqn_bidder.replay_buffer.append(([x[-1], marginal_price, Q], ind, payoff_action, [x_tmp[-1], marginal_price_tmp, Q]))
            bidder.history_payoff_profile.append(np.array(payoffs_each_action))
            regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff)) / (t + 1)
            for i in range(dqn_bidder.training_epochs):
                bidder.update_weights()

            if t % dqn_bidder.target_update_freq == 0:
                dqn_bidder.update_target_model()

            game_data.regrets.append([regret])

            # store data
            game_data.Q.append(Q)
            game_data.SW.append(social_welfare)
            game_data.bids.append(bids)
            game_data.bids_ind.append(inds)
            game_data.allocations.append(x)
            game_data.payments.append(payments)
            game_data.marginal_prices.append(marginal_price)

        player_final_dists.append(dqn_bidder.weights)
        game_data_profile[0].append(game_data)

    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
        pickle.dump(player_final_dists, file)

def Trustful_vs_Hedge(num_games, num_runs, T, file_name, seed=20, static_profiles=False, welfare=False):
    types = ['Trustful vs Hedge']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_cost_Hedge = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095] #last player
    d_cost_Hedge = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]
 
    
    # Actions of others obtained from diagonalization + their true cost
    HG_profile = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000
    
    
    hedge_bidder = Hedge_bidder(c_cost_Hedge, d_cost_Hedge, K, max_payoff, T)
    
    player_final_dists = []
    for run in tqdm(range(num_runs)):
        hedge_bidder.restart()
        game_data = auction_data()

        for t in range(T):
            if static_profiles:
                Q = Q
                cap = cap
            else:
                cap = 700
                cap = [np.random.normal(loc=cap, scale=0.15*cap, size=T) for _ in range(5)]
                cap = [cap[i][t] for i in range(5)]
                Q = 1448.4
                np.random.seed(seed)
                Q_load = np.random.normal(loc=Q, scale=0.15*Q, size=T)
                Q = Q_load[t]
            action, ind = hedge_bidder.choose_action()
#             action, ind = (0.02, 12), 5
            hedge_bidder.played_action = action
            hedge_bidder.history_action.append(ind)
            bids = HG_profile + [action]
            if welfare:
                x, marginal_price, payments, social_welfare = optimize_welfare(bids, cap)
            else:
                x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)
            
            payoff = []
            for i in range(N):
                if i == N-1:
                    payoff_HG = payments[-1] - (0.5 * hedge_bidder.cost[0] * x[-1] + hedge_bidder.cost[1]) * x[-1]
                    hedge_bidder.history_payoff.append(payoff_HG)
                    payoff.append(payoff_HG)
#                     if t==T-1:
#                         print(f'Run {run}) Hedge player payoff at round {t+1}: {payoff_HG}')
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)



            bidder = hedge_bidder
            i = -1
            payoffs_each_action = []
            for j, action in enumerate(bidder.action_set):
                tmp_bids = bids.copy()
                tmp_bids[i] = action
                if welfare:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_welfare(tmp_bids, cap)
                else:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                payoffs_each_action.append(payoff_action)
                bidder.cum_each_action[j] += payoff_action
            bidder.history_payoff_profile.append(np.array(payoffs_each_action))
            regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff))/(t+1)
            bidder.update_weights(bidder.history_payoff_profile[-1])

            game_data.regrets.append([regret])

            # store data
            game_data.Q.append(Q)
            game_data.SW.append(social_welfare)
            game_data.bids.append(bids)
            game_data.allocations.append(x)
            game_data.payments.append(payments)
            game_data.marginal_prices.append(marginal_price)
        player_final_dists.append(hedge_bidder.weights)
        game_data_profile[0].append(game_data)

    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
        pickle.dump(player_final_dists, file)
        
    ## Case b ==> Trustful Vs Random (others play Trustful vs B5 Plays Random)

# Case b ==> Trustful Vs Random
def Trustful_vs_Random(num_games , num_runs, T, file_name, seed=20, static_profiles=False, welfare=False):
    types = ['Trustful vs Random']
    game_data_profile = [[]]
    Q = 1448.4
    N = 5
    K = 10
    c_cost_Random = [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095] 
    d_cost_Random = [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]

    Trustful_profile = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

    cap = [700, 700, 700, 700, 700]
    max_payoff = 36000
    
    
    Random_bidder = random_bidder(c_cost_Random, d_cost_Random, K)
    
#     player_final_dists = []
    for run in tqdm(range(num_runs)):
        Random_bidder.restart()
        game_data = auction_data()
        for t in range(T):
            if static_profiles:
                Q = Q
                cap = cap
            else:
                cap = 700
                cap = [np.random.normal(loc=cap, scale=0.15*cap, size=T) for _ in range(5)]
                cap = [cap[i][t] for i in range(5)]
                Q = 1448.4
                np.random.seed(seed)
                Q_load = np.random.normal(loc=Q, scale=0.15*Q, size=T)
                Q = Q_load[t]
            action, ind = Random_bidder.choose_action()
            Random_bidder.played_action = action
            Random_bidder.history_action.append(ind)
            bids = Trustful_profile + [action]
            if welfare:
                x, marginal_price, payments, social_welfare = optimize_welfare(bids, cap)
            else:
                x, marginal_price, payments, social_welfare = optimize_alloc(bids, Q, cap)
            
            payoff = []
            for i in range(N):
                if i == N-1:
                    payoff_Random = payments[-1] - (0.5 * Random_bidder.cost[0] * x[-1] + Random_bidder.cost[1]) * x[-1]
                    Random_bidder.history_payoff.append(payoff_Random)
                    payoff.append(payoff_Random)
#                     if t==T-1:
#                         print(f'Run {run}) Random player payoff at round {t+1}: {payoff_Random}')
                else:
                    payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
                    payoff.append(payoff_bidder)
            game_data.payoffs.append(payoff)


            
            bidder = Random_bidder
            i = -1
            payoffs_each_action = []
            for j, action in enumerate(bidder.action_set):
                tmp_bids = bids.copy()
                tmp_bids[i] = action
                if welfare:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_welfare(tmp_bids, cap)
                else:
                    x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(tmp_bids, Q, cap)
                payoff_action = payments_tmp[i] - (0.5 * bidder.cost[0] * x_tmp[i] + bidder.cost[1]) * x_tmp[i]
                payoffs_each_action.append(payoff_action)
                bidder.cum_each_action[j] += payoff_action
            bidder.history_payoff_profile.append(np.array(payoffs_each_action))
            regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff))/(t+1)


            game_data.regrets.append([regret])

            # store data
            game_data.Q.append(Q)
            game_data.SW.append(social_welfare)
            game_data.bids.append(bids)
            game_data.allocations.append(x)
            game_data.payments.append(payments)
            game_data.marginal_prices.append(marginal_price)
        game_data_profile[0].append(game_data)


    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
## Case c ==> Hedge vs Hedge (All play Hedge)
# Case c ==> Hedge vs Hedge 

def all_Hedge(num_games, num_runs, T, file_name):
    types = ['All Hedge']
    Q = 1448.4
    N = 5
    K = 10
    c_limit = 0.08
    d_limit = 10
    c_list = [ 
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]
    
    d_list = [ 
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]]
    
    cap = [700, 700, 700, 700, 700]
    

    max_payoff = 36000

    types = []
    types.append('All Hedge')
    #types.append('Random')

    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            other_bidders.append(Hedge_bidder(c_list[i], d_list[i], K, max_payoff, T, has_seed = True))
            
        for type_idx, bidder_type in enumerate(types):
            if bidder_type == 'All Hedge':
                bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
#             if bidder_type == 'Random':
#                 bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
                
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True, regret_all=True))
                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
#         pickle.dump(player_final_dists, file)

   ## Case d ==> Hedge vs Random (Others play Hedge vs B5 plays Random)

# Case c ==> Hedge vs Hedge 

def Hedge_vs_Random(num_games, num_runs, T, file_name):

    Q = 1448.4
    N = 5
    K = 10
    c_limit = 0.08
    d_limit = 10
    c_list = [ 
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]
    
    d_list = [ 
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]]
    
    cap = [700, 700, 700, 700, 700]
    

    max_payoff = 36000

    types = []
    #types.append('All Hedge')
    types.append('Hedge vs Random')

    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            other_bidders.append(Hedge_bidder(c_list[i], d_list[i], K, max_payoff, T))
            
        for type_idx, bidder_type in enumerate(types):
#             if bidder_type == 'All Hedge':
#                 bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
            if bidder_type == 'Hedge vs Random':
                bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
                
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True, regret_all=True))
                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)

    ## Case e ==> Random vs Hedge (others play Random and B5 plays Hedge)

# Case e ==> Random vs Hedge 

def Random_vs_Hedge(num_games, num_runs, T, file_name):
    Q = 1448.4
    N = 5
    K = 10
    c_limit = 0.08
    d_limit = 10
    c_list = [ 
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]
    
    d_list = [ 
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]]
    
    cap = [700, 700, 700, 700, 700]
    

    max_payoff = 36000

    types = []
    types.append('Random vs Hedge')
#     types.append('Random vs Random')


    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
           # other_bidders.append(random_bidder(c_list[i], d_list[i], K, has_seed = True))
            other_bidders.append(random_bidder(c_list[i], d_list[i], K))
            
        for type_idx, bidder_type in enumerate(types):
            if bidder_type == 'Random vs Hedge':
                bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
#             if bidder_type == 'Random':
#                 bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
            
                
            player_final_dists = []
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True))

                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
#         pickle.dump(player_final_dists, file)

        
## Case f ==> Random vs Random (All play Random)
# Case f ==> Random vs Random 

def Random_vs_Random(num_games, num_runs, T, file_name):
    Q = 1448.4
    N = 5
    K = 10
    c_limit = 0.08
    d_limit = 10
    c_list = [ 
    [0.07, 0.08, 0.09, 0.10, 0.12, 0.075, 0.085, 0.095, 0.15, 0.17], 

    [0.02, 0.05, 0.06, 0.15, 0.25, 0.025, 0.15, 0.90, 0.13, 0.31], 

    [0.03, 0.04, 0.06, 0.12, 0.14, 0.095, 0.08, 0.09, 0.21, 0.27],
    
    [0.008, 0.01, 0.075, 0.24, 0.31,0.08, 0.09, 0.05, 0.11, 0.14],
    
    [0.01, 0.09, 0.10, 0.21, 0.97, 0.020, 0.13, 0.075, 0.19, 0.095]]
    
    d_list = [ 
    [9, 10, 11.5, 14, 13, 10, 12, 13, 15, 11], 

    [10, 15, 12, 17, 11, 12, 14, 13, 65, 14], 

    [12, 14, 13, 16, 15, 13, 12, 15, 14, 12],
    
    [12, 14, 17, 15, 17, 14, 11, 17, 15, 12],

    [11, 13, 11, 17, 20, 12, 11, 15, 17, 20]]
    
    cap = [700, 700, 700, 700, 700]
    

    max_payoff = 36000

    types = []
    types.append('Random vs Random')
#     types.append('Random')


    game_data_profile = [[] for i in range(len(types))]
    for j in range(num_games):
        other_bidders = []
        for i in range(N - 1):
            #other_bidders.append(random_bidder(c_list[i], d_list[i], K, has_seed = True))
            other_bidders.append(random_bidder(c_list[i], d_list[i], K))
            
        for type_idx, bidder_type in enumerate(types):
            if bidder_type == 'Random vs Random':
#                 bidders = other_bidders + [Hedge_bidder(c_list[-1], d_list[-1], K, max_payoff, T)]
#             if bidder_type == 'Random':
                bidders = other_bidders + [random_bidder(c_list[-1], d_list[-1], K)]
            
                
            player_final_dists = []
            for run in tqdm(range(num_runs)):
                game_data_profile[type_idx].append(run_auction(T, bidders, Q, cap, regret_calc=True))

                
    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)

if __name__ == "__main__":
    seed = np.random.randint(0, 100)
    #Trustful_vs_DDPG(num_games = 1, num_runs = 15, T = 200, file_name='TrustfulDDPG', seed = seed)
    Trustful_vs_DQN(num_games = 1, num_runs = 15, T = 200, file_name='TrustfulDQN', seed = seed)
    Trustful_vs_Hedge(num_games = 1, num_runs = 15, T = 200, file_name='TrustfulHG', seed = seed)
    Trustful_vs_Random(num_games = 1, num_runs = 15, T = 200, file_name ='TrustfulRandom', seed = seed)

    #all_Hedge(num_games = 1, num_runs = 15, T = 200, file_name ='allHG')
    #Hedge_vs_Random(num_games = 1, num_runs = 15, T = 200, file_name = 'Hedge_vs_Random')
    #Random_vs_Hedge(num_games = 1, num_runs=15, T=200, file_name='Random_Hedge')
    #Random_vs_Random(num_games = 1, num_runs = 15, T=200, file_name='all_Random')