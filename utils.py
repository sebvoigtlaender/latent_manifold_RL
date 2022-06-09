# utilities skeleton
# import section
import argparse
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Union
import numpy as np
import torch as pt
import torch.nn.functional as F

from absl import logging

from global_arguments import get_args

ListType = Union[List, np.ndarray]
TensorType = Union[pt.Tensor, pt.LongTensor]
TensorDict = MutableMapping[str, TensorType]

logging.set_verbosity(logging.INFO)


def default_config() -> MutableMapping[str, Any]:
    parser = argparse.ArgumentParser()
    config = parser.parse_args(args=[])
    return config

def env_config(args: MutableMapping[str, Any],
                              config: MutableMapping[str, Any],
                              env_name: str) -> None:
    '''extend default configuration in-place'''
    config.batch_order = 'time_first'
    if not hasattr(config, 'batch_size'):
        config.batch_size = args.batch_size
    config.device = args.device

    if env_name == 'toggle':
        config.env_name = env_name
        logging.info(f'environment = {env_name}')
        config.n_actions = 3
        config.d_cue = 1
        config.len_cue_sequence = 10
        config.n_cues = 2

    elif env_name == 'square_full_access':
        config.env_name = env_name
        logging.info(f'environment = {env_name}')
        config.height = 5
        config.initial_idx = (0,)
        config.n_actions = config.height**2
        config.n_steps = 20

    elif env_name == 'square_with_walls':
        config.env_name = env_name
        logging.info(f'environment = {env_name}')
        config.height = 5
        config.initial_idx = (0, 0)
        config.n_actions = 4
        config.n_steps = 20

def model_config(args: MutableMapping[str, Any],
                        config: MutableMapping[str, Any],
                        env: Optional[Any] = None) -> None:
    '''extend default configuration in-place. When merging the configurations take care that arguments agree'''
    if not hasattr(config, 'batch_size'):
        config.batch_size = args.batch_size

    assert hasattr(config, 'env_name'), 'specify environment before setting the network parameters'
    config.n_hidden_neurons = 20
    config.n_output_neurons = config.n_actions
    config.n_layers = 1
    if config.env_name == 'toggle':
        config.n_u = env.observation_space.shape[0]
        config.n_state_neurons = env.observation_space.shape[0]
    elif config.env_name == 'square_full_access':
        config.n_u = env.observation_space.shape[0]
        config.n_state_neurons = 0
    elif config.env_name == 'square_with_walls':
        config.n_u = int(env.config.height**2)
        config.n_state_neurons = 0
    config.n_input_neurons = config.n_u + config.n_state_neurons

    config.act_fn = 'relu'
    config.discrete = True
    config.k_B = 1.0
    config.stochastic = True

def merge_configurations() -> MutableMapping[str, Any]:
    pass

def activation_fn(act_fn: str) -> Callable[[TensorType], TensorType]:
    if act_fn == 'elu':
        return F.elu
    if act_fn == 'gelu':
        return F.gelu
    if act_fn == 'id':
        return pt.nn.Identity()
    elif act_fn == 'relu':
        return F.relu
    elif act_fn == 'tanh':
        return pt.tanh
    if act_fn == 'sigmoid':
        return pt.sigmoid
    else:
        raise NotImplementedError(act_fn)


def loss_fn(loss_fn: str) -> Callable[[TensorType], TensorType]:
    if loss_fn == 'mse_loss':
        return F.mse_loss
    if loss_fn == 'binary_cross_entropy':
        return F.binary_cross_entropy
    if loss_fn == 'kl_div':
        return F.kl_div
    else:
        raise NotImplementedError(loss_fn)

def __add__(tuple_1, tuple_2):
    return tuple(map(lambda x, y: x + y, tuple_1, tuple_2))

def calculate_eps(t: int) -> int:
    epsilon = 1 - np.clip(1 - 1e-4 * t, 0, 1)
    return epsilon

def convert_to_tensor(x: List, device: str = 'cpu') -> TensorType:
    x = pt.Tensor(x).to(device)
    return x

def join_tokens(*x, dim: int = -1, mode = 'cat'):
    if mode == 'cat':
        tokens = pt.cat([*x], dim)
    else:
        raise NotImplementedError()
    return tokens


def sync_grad(t: int, t_sync: int, model: pt.nn.Module, model_target: pt.nn.Module) -> None:
    
    if t % t_sync == 0:
        model_target.load_state_dict(model.state_dict())


class Replay_Buffer():

    def __init__(self, u_size: int, x_size: int, capacity: int = int(1e3)) -> None:
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.u = pt.zeros(capacity, u_size)
        self.x = pt.zeros(capacity, x_size)
        self.d_loss = pt.zeros(capacity, 1)
        self.dx = pt.zeros(capacity, x_size)
        self.done = pt.zeros(capacity, 1)

    def store(self, u, x, d_loss, dx, done):
        self.u[self.idx] = u
        self.x[self.idx] = x
        self.d_loss[self.idx] = d_loss
        self.dx[self.idx] = dx
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def spl(self, batch_size):
        idcs = rnd.randint(0, self.size, size=batch_size)
        u = pt.FloatTensor(self.u[idcs]).to(device)
        x = pt.FloatTensor(self.x[idcs]).to(device)
        d_loss = pt.FloatTensor(self.d_loss[idcs]).to(device)
        dx = pt.FloatTensor(self.dx[idcs]).to(device)
        done = pt.FloatTensor(self.done[idcs]).to(device)
        return u, x, d_loss, dx, done

# epsilon = lambda t: 0.05 + (0.05 - 1) * np.exp(-t / decay)
