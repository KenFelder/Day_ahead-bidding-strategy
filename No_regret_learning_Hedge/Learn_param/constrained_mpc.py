#!/usr/bin/env python
# coding: utf-8

# # Constrained MPC
# 
# This notebook accompanies the paper [Learning Convex Optimization Models](https://web.stanford.edu/~boyd/papers/learning_copt_models.html).

# In[1]:


import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm

from cvxpylayers.torch import CvxpyLayer


get_ipython().run_line_magic('matplotlib', 'inline')

torch.set_default_dtype(torch.double)


# In[2]:


n = 10
m = 4
val_seed = 243
np.random.seed(0)
torch.manual_seed(0)
A_np = np.random.randn(n, n)
A_np /= np.max(np.abs(np.linalg.eig(A_np)[0]))
B_np = np.random.randn(n, m)

A = torch.tensor(A_np)
B = torch.tensor(B_np)
weights = torch.randn(n).abs()
weights_np = weights.numpy()
beta = 0.5
T = 5


def dynamics(xt, ut):
    return A @ xt + B @ ut + torch.randn(xt.shape)

def stage_cost(xt, ut):
    cost = 0.    
    return (weights*(xt.pow(2))).sum() + ut.pow(2).sum()


# In[3]:


def construct_mpc_problem():
    x = cp.Parameter(n)
    states = [cp.Variable(n) for _ in range(T)]
    controls = [cp.Variable(m) for _ in range(T)]
    constraints = [states[0] == x, cp.norm(controls[0], 'inf') <= beta]
    objective = cp.sum(cp.multiply(weights_np, cp.square(states[0]))) + cp.sum_squares(controls[0])
    for t in range(1, T):
        objective += cp.sum(cp.multiply(weights_np, cp.square(states[t]))) + cp.sum_squares(controls[t])
        constraints += [states[t] == A_np @ states[t-1] + B_np @ controls[t-1]]
        constraints += [cp.norm(controls[t], 'inf') <= beta] 
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, variables=[controls[0]], parameters=[x])

mpc_policy = construct_mpc_problem()


# In[4]:


def simulate(policy, n_iters=1000, seed=0):
    torch.random.manual_seed(seed)

    x0 = torch.randn(n)
    states = [x0]
    controls = []
    costs = []
    for t in tqdm(range(n_iters)):
        xt = states[-1]
        ut = policy(xt)[0]
        controls.append(ut)
        costs.append(stage_cost(xt, ut).item())
        states.append(dynamics(xt, ut))
    return states[:-1], controls, costs
    
states, controls, costs = simulate(mpc_policy)
scale = 0.31622776601
torch.manual_seed(1)
for control in controls:
    control.add_(scale*torch.randn(control.shape))
    control.clamp_(-beta, beta)


_, val_mpc_controls, val_mpc_costs = simulate(mpc_policy, seed=val_seed)
for control in val_mpc_controls:
    control.add_(scale*torch.randn(control.shape))
    control.clamp_(-beta, beta)


# In[5]:


def mse(preds, actual):
    preds = torch.stack(preds, dim=0)
    actual = torch.stack(actual, dim=0)
    return (preds - actual).pow(2).mean(axis=1).mean(axis=0).item()


# In[6]:


_, true_copt_controls, _ = simulate(mpc_policy, seed=val_seed)
true_mse = mse(true_copt_controls, val_mpc_controls)
true_mse


# In[7]:


def construct_agent_mpc_problem():
    x = cp.Parameter(n)
    weights = cp.Parameter(n, nonneg=True)
    states = [cp.Variable(n) for _ in range(T)]
    controls = [cp.Variable(m) for _ in range(T)]
    constraints = [states[0] == x, cp.norm(controls[0], 'inf') <= beta]
    objective = cp.sum(cp.multiply(weights, cp.square(states[0]))) + cp.sum_squares(controls[0])
    for t in range(1, T):
        objective += cp.sum(cp.multiply(weights, cp.square(states[t]))) + cp.sum_squares(controls[t])
        constraints += [states[t] == A_np @ states[t-1] + B_np @ controls[t-1]]
        constraints += [cp.norm(controls[t], 'inf') <= beta] 
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, variables=[controls[0]], parameters=[x, weights])


# In[ ]:


adp_policy = construct_agent_mpc_problem()

weights_tch = torch.ones(n, requires_grad=True)
state_feedback_policy = lambda x: adp_policy(x, weights_tch)

epochs = 20
val_losses = []
losses = []
with torch.no_grad():
    _, initial_preds, _ = simulate(state_feedback_policy, seed=val_seed)
    val_losses.append(mse(initial_preds, val_mpc_controls))
    print(val_losses[-1])
opt = torch.optim.Adam([weights_tch], lr=3e-4)

for epoch in range(epochs):
    print('Epoch: ', epoch)
    for xt, ut in tqdm(zip(states, controls)):
        opt.zero_grad()
        ut_hat = adp_policy(xt, weights_tch)[0]
        loss = (ut - ut_hat).pow(2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    with torch.no_grad():
        weights_tch.data = weights_tch.relu()
        _, pred_ctrls, _ = simulate(state_feedback_policy, seed=val_seed)
        val_losses.append(mse(pred_ctrls, val_mpc_controls))
    print(val_losses[-1])


# In[9]:


plt.plot(val_losses)


# In[10]:


state_feedback_policy = lambda x: adp_policy(x, weights_tch)
agent_states, agent_controls, agent_costs = simulate(state_feedback_policy)


# In[11]:


weights_np


# In[12]:


weights_tch.detach().numpy()


# In[13]:


def plot_weights(w, h):
    fig = plt.figure()
    fig.set_size_inches((w, h))
    plt.plot(weights_tch.detach().numpy(), linestyle='-', color='k', label='learned')
    plt.plot(weights_np, linestyle='--', color='k', label='true')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\theta_i$')
    plt.legend()
    plt.tight_layout()


# In[14]:


w, h = 10., 3.5
plot_weights(w, h)
plt.show()


# ### NN

# In[15]:


class FF(torch.nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=n, out_features=n)
        self.fc2 = torch.nn.Linear(in_features=n, out_features=m)
        
    def forward(self, x):
        h1 = self.fc1(x).relu()
        return self.fc2(h1).clamp(-beta, beta)


# In[ ]:


torch.random.manual_seed(0)
ff = FF()


epochs = 100
nn_losses = []
val_nn_losses = []
opt = torch.optim.Adam(ff.parameters(), lr=3e-4)
for epoch in range(epochs):
    print('Epoch: ', epoch)
    for xt, ut in tqdm(zip(states, controls)):
        opt.zero_grad()
        ut_hat = ff(xt)
        loss = (ut - ut_hat).pow(2).mean()
        loss.backward()
        nn_losses.append(loss.item())
        opt.step()
    _, val_preds, _ = simulate(lambda x: [ff(x)], seed=val_seed)
    with torch.no_grad():
        val_nn_losses.append(mse(val_preds, val_mpc_controls))
    print(val_nn_losses[-1])


# In[17]:


val_losses[-1]


# In[18]:


val_nn_losses[-1]


# In[19]:


plt.plot(val_nn_losses)


# In[20]:


def plot_losses(w, h):
    fig = plt.figure()
    fig.set_size_inches((w, h))
    plt.axhline(val_nn_losses[-1], color='k', linestyle='-.', label='NN')    
    plt.plot(np.arange(len(val_losses)) + 1, val_losses, color='k', label='COM')
    plt.axhline(true_mse, color='k', linestyle='--', label='true')
    plt.xticks([1, 5, 10, 15, 20])
    plt.xlabel('iteration')
    plt.ylabel('validation loss')
    plt.legend(loc='upper right')
    plt.tight_layout()


# In[21]:


w, h = 10, 3.5
plot_losses(w, h)
plt.show()


# In[23]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


w = 10
h = 3.5

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(w, h)

ax[1].plot(weights_tch.detach().numpy(), linestyle='-', color='k', label='learned')
ax[1].plot(weights_np, linestyle='--', color='k', label='true')
ax[1].set_xlabel(r'$i$')
ax[1].set_ylabel(r'$\theta_i$')
ax[1].set_xticks([0, 2,4, 6, 8])
ax[1].legend()
    
ax[0].axhline(val_nn_losses[-1], color='k', linestyle='-.', label='NN')
ax[0].plot(np.arange(len(val_losses)) + 1, val_losses, color='k', label='COM')
ax[0].axhline(true_mse, color='k', linestyle='--', label='true')
ax[0].set_xticks([1, 5, 10, 15, 20])
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('validation loss')
ax[0].legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:




