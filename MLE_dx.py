# maze with finite step length = 1
# result replicated, bias = False and n_layers important
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
args.grad = 'sparse'
args.lr = 0.003
args.n_episodes = 50000

config = default_config()
env_config(args, config, 'square_with_walls')
config.height = 5
config.n_steps = 100
env = Maze(config)
model_config(args, config, env)
config.k_B = 1.0


config.stochastic = True

logging.info(args)
logging.info(config)

actor_critic = ActorCritic(config).to(args.device)
actor = actor_critic
optimizer = CoreOptimizer(actor.actor.parameters(), lr=args.lr, args=args)
actor.train()

u, done = env.reset()
log_p_action_list = []
p_total_path_list = []
losses = []
episode_return = 0
print(actor.actor)
for t in range(args.n_episodes):

    u = convert_to_tensor(u)
    p_action, action = actor.act(u)
    log_p_action_list.append(p_action.log_prob(action))

    u, reward, done = env.step(action)
    episode_return += reward

    if done:
        losses.append(episode_return)
        log_p_action = pt.sum(pt.stack(log_p_action_list, 0), 0)
        optimizer.zero_grad()
        log_p_action.backward()
        optimizer.step(scaling_factor = episode_return)

        if reward == 1.0:
            p_total_path_list.append(log_p_action.item())

        u, done = env.reset()

        episode_return = 0
        log_p_action_list = []


# printing
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(2, 1, 1)
ax.plot(losses, linewidth = 0.8)
ax = fig.add_subplot(2, 1, 2)
ax.plot(p_total_path_list, linewidth = 0.8)
plt.pause(10)
plt.close(fig)