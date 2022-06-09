# optimizer
from typing import Any, Callable, Mapping, MutableMapping, Iterable, Optional

import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from absl import logging

from utils import activation_fn, default_config, TensorDict, TensorType 

logging.set_verbosity(logging.INFO)


class CoreOptimizer(pt.optim.Optimizer):

    def __init__(self, parameters: Iterable,
                       lr: float,
                       args: Optional[MutableMapping[str, Any]] = None) -> None:
        if lr < 0.0:
            logging.info('Optimizer: gradient descend')
        elif lr > 0.0:
            logging.info('Optimizer: gradient ascend')
        elif lr == 0.0:
            raise ValueError(f'Learning rate is {lr}, no learning')
        defaults = dict(lr = lr)
        if args:
            self.args = args
        super(CoreOptimizer, self).__init__(parameters, defaults)

    def step(self, scaling_factor: pt.float32 = None, closure: Optional[Callable] = None) -> None:
        '''Performs a single optimization step'''
        with pt.no_grad():
            loss = None
            if closure is not None:
                with pt.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                p_with_grad = []
                d_p_list = []
                lr = group['lr']
                for p in group['params']:
                    if p.grad is not None:
                        p_with_grad.append(p)
                        d_p_list.append(p.grad)
                        state = self.state[p]
                for i, p in enumerate(p_with_grad):
                    d_p = d_p_list[i]
                    if scaling_factor is not None:
                        d_p.multiply_(scaling_factor) # in most cases this is loss.item()
                    p.add_(d_p, alpha = lr)
                for p in p_with_grad:
                    state = self.state[p]

    def accumulate_grad(self, p_actions: TensorType, idx: TensorType) -> None:
        if self.args.grad == 'sparse':
            g_p_actions = pt.zeros(p_actions.shape).to(self.args.device)
            g_p_actions[idx] = 1 # in some cases idx is simply -1
            pt.log(p_actions).backward(gradient = g_p_actions)
        elif self.args.grad == 'dense':
            g_p_actions = pt.ones(p_actions.shape).to(self.args.device)
            pt.log(p_actions).backward(gradient = g_p_actions)
        

class SwarmOptimizer(pt.optim.Optimizer):
    '''this optimizer was taken from the evolutionary approach policy swarm, it is different from the optimizers below'''
    def __init__(self, parameters: Iterable,
                       lr: float,
                       args: Optional[MutableMapping[str, Any]] = None) -> None:
        defaults = dict(lr = lr)
        if args:
            self.args = args
        super(SwarmOptimizer, self).__init__(parameters, defaults)

    def step(self, fitness: pt.float32 = None, closure: Optional[Callable] = None) -> None:
        '''Performs a single optimization step'''
        if not fitness:
            raise ValueError(f'expected an argument fitness, got None instead')
        with pt.no_grad():
            loss = None
            if closure is not None:
                with pt.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                p_with_grad = []
                lr = group['lr']
                for p in group['params']:
                    if p.grad is not None:
                        p_with_grad.append(p)
                        state = self.state[p]
                for i, p in enumerate(p_with_grad):
                    if not self.args.perturb:
                        p.add_(fitness, alpha = lr/(self.args.n_state_neurons * self.args.std))
                    if self.args.perturb:
                        p.add_(pt.full(p.size(), self.args.std), alpha = self.args.epsilon)
                        # p.add_(pt.randn(p.size()), alpha = self.lr)
                for p in p_with_grad:
                    state = self.state[p]