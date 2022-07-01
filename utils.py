# args changeable by user, config forces arguments according to the specifications given by the user
import argparse
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Union
import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn.functional as F

from absl import logging

from arguments import get_args

logging.set_verbosity(logging.INFO)

ListType = Union[List, np.ndarray]
TensorType = Union[pt.Tensor, pt.LongTensor]
TensorDict = MutableMapping[str, TensorType]


def activation_fn(act_fn: str = 'relu') -> Callable[[TensorType], TensorType]:
    '''
    Activation function, selected by keyword 'act_fn', default = relu
    '''
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

def fix_seed(args: MutableMapping[str, Any]) -> None:
    '''
    Global random seed
    '''
    if args.seed:
        rnd.seed(0)
        pt.manual_seed(0)
        
def loss_fn(loss_fn: str) -> Callable[[TensorType], TensorType]:
    '''
    Loss function, selected by keyword 'loss_fn'
    '''
    if loss_fn == 'mse_loss':
        return F.mse_loss
    if loss_fn == 'binary_cross_entropy':
        return F.binary_cross_entropy
    if loss_fn == 'kl_div':
        return F.kl_div
    else:
        raise NotImplementedError(loss_fn)

def __add__(tuple_1, tuple_2):
    '''
    Add method for tuples, behaves like vector addition
    '''
    return tuple(map(lambda x, y: x + y, tuple_1, tuple_2))

def calculate_eps(t: int) -> float:
    '''
    Perturbation noise parameter for evolution strategies, scaled by time t
    '''
    epsilon = 1 - np.clip(1 - 1e-4 * t, 0, 1)
    return epsilon

def convert_to_tensor(x: ListType, device: str = 'cpu') -> TensorType:
    '''
    Convert list or numpy array to tensor
    '''
    x = pt.Tensor(x).to(device)
    return x

def join_tokens(*x: TensorType, dim: int = -1, mode: str = 'cat'):
    '''
    Combine list of tensors along dimension dim, default mode concatenate
    '''
    if mode == 'cat':
        tokens = pt.cat([*x], dim)
    else:
        raise NotImplementedError()
    return tokens

def sync_grad(t: int, t_sync: int, model: pt.nn.Module, model_target: pt.nn.Module) -> None:
    '''
    Synchronize actor and actor target every t_sync iterations
    '''
    if t % t_sync == 0:
        model_target.load_state_dict(model.state_dict())


class Replay_Buffer():
    '''
    Replay buffer for reinforcement learning
    '''

    def __init__(self,
                 u_size: int,
                 x_size: int,
                 capacity: int = int(1e3)) -> None:

        '''
        idx is the position up to which the replay buffer is filled
        size is the current length of the buffer
        u is the container for external input
        x is the container for internal state input
        loss is the container for feedback signals from the environment, equivalent to reward
        dx is the container for state updates, equivalent to action
        done is the container for done signals from the environment
        '''
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.u = pt.zeros(capacity, u_size)
        self.x = pt.zeros(capacity, x_size)
        self.loss = pt.zeros(capacity, 1)
        self.dx = pt.zeros(capacity, x_size)
        self.done = pt.zeros(capacity, 1)

    def store(self, u, x, loss, dx, done):
        self.u[self.idx] = u
        self.x[self.idx] = x
        self.loss[self.idx] = loss
        self.dx[self.idx] = dx
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def spl(self, batch_size):
        idcs = rnd.randint(0, self.size, size=batch_size)
        u = pt.FloatTensor(self.u[idcs]).to(device)
        x = pt.FloatTensor(self.x[idcs]).to(device)
        loss = pt.FloatTensor(self.loss[idcs]).to(device)
        dx = pt.FloatTensor(self.dx[idcs]).to(device)
        done = pt.FloatTensor(self.done[idcs]).to(device)
        return u, x, loss, dx, done