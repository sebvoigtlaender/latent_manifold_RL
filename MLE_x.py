import argparse
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from absl import logging

from env import Maze
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
args.grad = 'sparse'
args.lr = 0.1
args.n_episodes = 1000

config = default_config()
env_config(args, config, 'square_full_access')
config.n_steps = 10
config.k_B = 0.1
env = Maze(config)
model_config(args, config, env)

config.stochastic = True

logging.info(args)
logging.info(config)

actor_critic = ActorCritic(config).to(args.device)
actor = actor_critic
optimizer = CoreOptimizer(actor.actor.parameters(), lr=args.lr, args=args)
actor.train()

# init
u, done = env.reset()
p_action_list = []
p_action_target_list = []
losses = []

# train loop
# for t in range(args.n_episodes):

#     u = convert_to_tensor(u)
#     p_action, action = actor.act(u)
#     p_action_list.append(p_action.probs)
#     u, reward, done = env.step(action)
#     losses.append(reward)

#     if done:
#         p_action_list = pt.sum(pt.stack(p_action_list, 0), 0)
#         optimizer.zero_grad()
#         optimizer.accumulate_grad(p_action_list, -1)
#         optimizer.step(scaling_factor = reward)
        
#         p_action_list = []
#         u, done = env.reset()

for t in range(args.n_episodes):

    u = convert_to_tensor(u)
    p_action, action = actor.act(u)
    p_action_list.append(p_action.probs[-1].item())
    u, reward, done = env.step(action)
    losses.append(reward)

    if done == 1:
        optimizer.zero_grad()
        optimizer.accumulate_grad(p_action.probs, action.item())
        optimizer.step(scaling_factor = reward)
        u, done = env.reset()


# printing

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(losses, linewidth = 0.8)
ax = fig.add_subplot(2, 1, 2)
ax.plot(p_action_list, linewidth = 0.8)
plt.show()
# plt.pause(3)
# plt.close(fig)