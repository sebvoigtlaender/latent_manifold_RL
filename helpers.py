import argparse
from typing import Any, Mapping, MutableMapping, Optional, Union

import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging

from arguments import get_args
from utils import activation_fn, join_tokens, TensorDict, TensorType 

logging.set_verbosity(logging.INFO)


def get_p_action_target(args: MutableMapping[str, Any],
                        x: TensorType,
                        state_target: TensorType,
                        transition_table: TensorType) -> TensorType:

    '''
    If transition_table is given, calculate target for distribution over actions or state updates dx

    Args: 
        args: non-tunable hyperparameters
        x: latent state
        state_target: target latent state
        transition_table: tensor containing all possible transitions dx

    Returns:
       p_action_target: probability distribution over all possible actions with mode at the locally optimal transition  
    '''

    with pt.no_grad(): 
        dx = pt.clamp(state_target - x, min=transition_table[0], max=transition_table[-1])
        target_idx = pt.where(transition_table == dx)[1]
        p_action_target = pt.zeros(args.batch_size, args.n_actions).to(args.device)
        p_action_target[range(args.batch_size), target_idx] = 1
        
    return p_action_target


def value_act(args: MutableMapping[str, Any],
          u: TensorType,
          x: TensorType, 
          state_target: TensorType,
          transition_table: TensorType,
          epsilon: int, 
          mode: Optional[str] = 'min') -> pt.LongTensor:

    '''
    If transition_table is given, calculate target state update dx
    based on the locally optimal transition as measured by state value

    Args: 
        args: non-tunable hyperparameters
        u: external input
        x: latent state
        state_target: target latent state
        transition_table: tensor containing all possible transitions dx
        epsilon: probability of random exploration
        mode: keyword that characterizes if optimal index is to be found by minimizing
              or maximizing or else, currently not implemented

    Returns: 
        action_idx: index used to pick action dx from transition table
    '''

    with pt.no_grad():
        action_idx = pt.zeros(args.batch_size)

        dx = pt.clamp(state_target - x, min=transition_table[0], max=transition_table[-1])
        opt_idx = pt.where(transition_table == dx)[1] # locally optimal index

        assert (x + transition_table).shape == (args.batch_size, args.n_actions)
        x = join_tokens(u.expand((args.batch_size, args.n_actions)), (x + transition_table))
        values = actor_critic.state_value(x.view(*x.shape, 1))
        value_idx = pt.min(values, 1, False).indices # index chosen based on state value

        non_opt_idx = rnd.rand(args.batch_size) > epsilon # mix indices based on proportion given by epsilon
        action_idx = opt_idx
        action_idx[non_opt_idx] = value_idx

    return action_idx


def train_actor_critic(args: Mapping[str, Any], 
                       replay_buffer: Any) -> None:

    '''
    Basic actor critic training algorithm

    Args: 
        args: non-tunable hyperparameters
        replay_buffer: basic reinforcement learning replay buffer
    '''

    u, x, d_loss, dx, done = replay_buffer.spl(args.batch_size)

    q = actor_critic.q(u, x)
    target_q = d_loss + (args.discount * (1 - done) * actor_critic_target.q(u, x + actor_critic_target.act(u, x))).detach()
    critic_loss = F.mse_loss(q, target_q)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    actor_loss = -actor_critic.q(u, x + actor_critic.act(u, x)).mean()
    p_opt.zero_grad()
    actor_loss.backward()
    p_opt.step()

    for p, p_target in zip(actor_critic.parameters(), actor_critic_target.parameters()):
        p_target.data.copy_(args.tau * p.data + (1 - args.tau) * p_target.data)