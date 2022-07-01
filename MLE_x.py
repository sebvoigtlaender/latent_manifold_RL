import argparse
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging

from arguments import get_args
from env import get_env_config, get_env, Toggle, Maze
from helpers import get_p_action_target, value_act
from model import get_actor_critic_config, ActorCritic, Reservoir
from optimizer import CoreOptimizer
from utils import activation_fn, convert_to_tensor, join_tokens, TensorDict, TensorType 

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.style.use('seaborn-poster')
plt.style.use('seaborn-white')


logging.set_verbosity(logging.INFO)

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
args = get_args()
args.env_id = 'square_full_access'
args.lr = 0.1
args.n_episodes = 10000
args.stochastic = True


env_config = get_env_config(args)
actor_critic_config = get_actor_critic_config(args)

logging.info(f'hyperp: {args}')
logging.info(f'env_config: {env_config}')
logging.info(f'actor_critic_config: {actor_critic_config}')

env = get_env(env_config)
actor_critic = ActorCritic(actor_critic_config).to(args.device)
optimizer = CoreOptimizer(actor_critic.actor.parameters(), lr=args.lr, args=args)
actor_critic.train()

# init
u, done = env.reset()
p_action_list = []
p_action_target_list = []
losses = []

# train loop
for t in range(args.n_episodes):

    u = convert_to_tensor(u)
    p_action, action = actor_critic.act(u)
    p_action_list.append(p_action.probs)
    u, reward, done = env.step(action)
    losses.append(reward)

    if done:
        p_action_list = pt.sum(pt.stack(p_action_list, 0), 0)
        optimizer.zero_grad()
        optimizer.accumulate_grad(p_action_list, -1)
        optimizer.step(scaling_factor = reward)
        
        p_action_list = []
        u, done = env.reset()

# for t in range(args.n_episodes):

#     u = convert_to_tensor(u)
#     p_action, action = actor_critic.act(u)
#     print(p_action.probs.shape)
#     u, reward, done = env.step(action)
#     losses.append(reward)

#     if done == 1:
#         optimizer.zero_grad()
#         optimizer.accumulate_grad(p_action.probs, action.item())
#         optimizer.step(scaling_factor = reward)
#         u, done = env.reset()


# printing

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(losses, linewidth = 0.8)
ax = fig.add_subplot(2, 1, 2)
# ax.plot(p_action_list, linewidth = 0.8)
plt.show()
# plt.pause(3)
# plt.close(fig)