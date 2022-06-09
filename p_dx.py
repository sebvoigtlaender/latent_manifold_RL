import argparse
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging

from env import Toggle
from global_arguments import get_args
from helpers import get_p_action_target, value_act
from model import ActorCritic, Reservoir
from optimizer import CoreOptimizer
from utils import default_config, env_config, model_config, activation_fn, convert_to_tensor, join_tokens, TensorDict, TensorType 

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.style.use('seaborn-poster')
plt.style.use('seaborn-white')


logging.set_verbosity(logging.INFO)

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
args = get_args()
args.batch_size = 16
args.lr = 0.001
args.n_episodes = 5000

config = default_config()
env_config(args, config, 'toggle')
env = Toggle(config)
model_config(args, config, env)

config.n_layers = 1
config.stochastic = False


logging.info(args)
logging.info(config)

transition_table = pt.Tensor([-1, 0, 1])
assert config.n_actions == len(transition_table)

actor_critic = ActorCritic(config).to(args.device)
actor = actor_critic
actor_target = ActorCritic(config).actor.to(args.device)
actor_target.load_state_dict(actor.state_dict(), strict=False)
reservoir = Reservoir(args, config, transition_table).to(args.device)
optimizer = pt.optim.Adam(actor.actor.parameters(), lr=args.lr)
actor.train()

# init
u, state_target, done = env.reset()
x = reservoir.clear()

p_action_list = []
p_action_target_list = []
losses = []

print(env.action_space)

# train loop
for t in range(args.n_episodes):

    u, state_target = convert_to_tensor(u), convert_to_tensor(state_target)
    p_action, action = actor.act(join_tokens(u, x))

    p_action_target = get_p_action_target(config, x, state_target, transition_table)

    x, state_transition = reservoir.update(x, idx = action)
    p_action_list.append(p_action.probs)
    p_action_target_list.append(p_action_target)

    u, state_target, done = env.step()

    if done:
        p_action_list = pt.stack(p_action_list, 0)
        p_action_target_list = pt.stack(p_action_target_list, 0)
        loss = F.kl_div(pt.log(p_action_list), p_action_target_list, reduction='batchmean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        p_action_list = []
        p_action_target_list = []
        u, state_target, done = env.reset()
        x = reservoir.clear()

print(u.flatten())

plt.figure(figsize=(8, 4))
plt.plot(losses, linewidth=0.8)
plt.yscale('log')
plt.show()







































#     seq = pt.Tensor(seq).transpose(0, 1).to(device)
#     target = pt.Tensor(target).transpose(0, 1).to(device)

#     return seq, target


# n_barcodes = 5
# T = 20
# b = 0.3
# k_B = 1
# n_cues = 5
# n_groups = 1
# n_heads = 1
# n_layers = 3
# _dx = np.array([-1, 0, 1])
# n_neurons = n_barcodes + 1
# n_x_neurons = n_barcodes
# cue_size = 10
# batch_size = 4
# out_size = 1
# h_size = 20
# device = 'cpu'
# lr = 0.01
# n_epochs = 10000

# barcodes = np.stack([rnd.choice([-1, 1], cue_size) for i in range(n_barcodes)])
# actor = Trigger(n_x_neurons + cue_size, h_size, n_neurons, n_layers).to(device)
# reservoir = reservoir(batch_size, n_x_neurons)
# optimizer = pt.optim.SGD(actor.parameters(), lr=lr)
# actor.train()
# losses = []
# x_losses = []

# for epoch in range(1, n_epochs+1):
    
#     x = pt.zeros((batch_size, n_x_neurons)).to(device)   
#     spike_sequence, target = generate_cue(T, batch_size, n_cues, cue_size, n_barcodes, barcodes)
#     diff_x_t_lst = pt.diff(pt.cat([pt.zeros(1, batch_size, n_barcodes).to(device), target]), dim=0)
#     p_dx_lst = []
#     p_dx_t_lst = []
#     reservoir_sequence = []
#     for t in range(T):
#         u = spike_sequence[t]
#         x_t = target[t]
#         diff_x_t = diff_x_t_lst[t]
#         p_dx = pt.softmax(actor(pt.cat([x, u], -1))/k_B, -1)
#         # p_dx = pt.softmax(actor(u)/k_B, -1)
#         p_dx_t = reservoir.p_dstate_target(x, x_t, diff_x_t)
#         x = reservoir.update_x(x, p_dx)
#         reservoir_sequence.append(pt.clone(x[0]))
#         p_dx_lst.append(p_dx)
#         p_dx_t_lst.append(p_dx_t)
#     p_dx_lst = pt.stack(p_dx_lst, 0)
#     p_dx_t_lst = pt.stack(p_dx_t_lst, 0)
#     p_dx_loss = F.kl_div(pt.log(p_dx_lst), p_dx_t_lst, reduction='batchmean')
#     optimizer.zero_grad()
#     p_dx_loss.backward()
#     optimizer.step()
#     losses.append(p_dx_loss.item())
#     x_losses.append(F.mse_loss(pt.stack(reservoir_sequence, 0), target[:, 0]).item())
    
#     if epoch % 10 == 0:
#         reservoir_sequence = pt.stack(reservoir_sequence, 0).cpu().numpy()

#         clear_output(True)
#         plt.figure(figsize=(20, 20))
#         plt.subplot(411)
#         plt.imshow(reservoir_sequence.T, cmap='gray')
#         plt.subplot(412)
#         plt.imshow(target[:, 0].cpu().numpy().T, cmap='gray')
#         plt.subplot(413)
#         plt.plot(losses, linewidth=1)
#         plt.yscale('log')
#         plt.subplot(414)
#         plt.plot(x_losses, linewidth=1)
#         # plt.yscale('log')
#         plt.show()


# p = rnd.rand(config.n_state_neurons)
# losses = []
# all_rewards = []
# x = pt.zeros(config.batch_size, config.n_state_neurons).to(args.device)

# for t in range(1, args.n_epochs+1):
#     q_lst = []
#     return_lst = []
#     x_updates = []
#     for n in range(args.n_env_steps):
#         p_actions, idx = actor.act(x)
#         q_lst.append(p_actions.probs[range(config.batch_size), idx])
#         dx = env.step(idx, p)
#         print(x.shape)
#         print('dx ', dx.shape)
#         all_rewards.append(dx.mean())
#         x = reservoir.update(x, dx, idx)
#         x_updates.append(pt.Tensor(np.float32(dx)))
#     q_t = q_net_t(x).max(1).values.detach()
#     for n in reversed(range(n_env_steps)):
#         q_t = x_updates[n] + g*q_t
#         return_lst.append(q_t)
#     q_lst = pt.stack(q_lst, 0)
#     return_lst = pt.stack(return_lst, 0)
#     loss = F.mse_loss(q_lst, return_lst)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
    
#     sync_grad(t)
    
#     if t % 100 == 0:
#         clear_output(True)
#         plt.figure(figsize=(20, 5))
#         plt.subplot(131)
#         plt.plot(p, '.', markersize=4)
#         plt.subplot(132)
#         plt.plot(np.convolve(all_rewards, np.ones(10)/10, mode='valid')[::10], '.', markersize=0.7)
#         plt.subplot(133)
#         plt.title('loss')
#         plt.plot(losses, '.', markersize=0.7)
#         plt.yscale('log')
#         plt.show()