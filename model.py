# model skeleton
# import section
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch as pt
import torch.nn as nn
from torch.distributions import Categorical

from absl import logging

from utils import activation_fn, default_config, TensorDict, TensorType 

logging.set_verbosity(logging.INFO)


class Core(pt.nn.Module):

    def __init__(self,
                 n_input_neurons: int,
                 n_hidden_neurons: int,
                 n_output_neurons: int, 
                 n_layers: int,
                 act_fn: str = 'relu',
                 bias: bool = True) -> None:
        super().__init__()
        self.act_fn = act_fn
        self.input_layer = pt.nn.Linear(n_input_neurons, n_hidden_neurons, bias = bias)
        self.hidden_layers = pt.nn.ModuleList([pt.nn.Linear(n_hidden_neurons, n_hidden_neurons, bias = bias) for i in range(n_layers)])
        self.output_layer = pt.nn.Linear(n_hidden_neurons, n_output_neurons, bias = bias)
        
    def forward(self, x: TensorType) -> TensorType:
        x = self.input_layer(x)
        x = activation_fn(self.act_fn)(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = activation_fn(self.act_fn)(x)
        x = self.output_layer(x)
        return x 


class ActorCritic(pt.nn.Module):

    def __init__(self, config: MutableMapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        if config.stochastic:
            self.actor = Core(config.n_input_neurons, config.n_hidden_neurons, config.n_output_neurons, config.n_layers)
            self.critic = Core(config.n_input_neurons, config.n_hidden_neurons, config.n_output_neurons, config.n_layers)
        elif not self.config.stochastic:
            if self.config.discrete:
                self.actor = Core(config.n_input_neurons, config.n_hidden_neurons, config.n_output_neurons, config.n_layers)
                self.critic = Core(config.n_input_neurons, config.n_hidden_neurons, config.n_output_neurons, config.n_layers)
            if not self.config.discrete:
                self.actor = Core(config.n_input_neurons, config.n_hidden_neurons, 1, config.n_layers)
                self.critic = Core(config.n_input_neurons, config.n_hidden_neurons, 1, config.n_layers)
        self.state_value = Core(config.n_input_neurons, config.n_hidden_neurons, 1, config.n_layers)
        

    def act(self, x: TensorType) -> Union[Tuple[TensorType, TensorType], TensorType]:
        if self.config.stochastic:
            p_action = Categorical(pt.softmax(self.actor(x)/self.config.k_B, -1))
            action = p_action.sample()
            return p_action, action
        elif not self.config.stochastic:
            if self.config.discrete:
                p_action = Categorical(pt.softmax(self.actor(x)/self.config.k_B, -1))
                action = p_action.probs.max(-1).indices
                return p_action, action
            else:
                action = pt.tanh(self.actor(x))
                return action

    def calc_q(self, x: TensorType, act_fn: Optional[str] = 'id') -> TensorType:
        q = self.critic(x)
        q = activation_fn(act_fn)(q)
        return q

    def calc_state_value(self, x: TensorType, act_fn: Optional[str] = 'id') -> TensorType:
        value = self.state_value(x)
        value = activation_fn(act_fn)(value)
        return value

    def q_act(self):
        pass


class Reservoir(pt.nn.Module):
    
    '''
    Reservoir performs the state update. In the state transition contains the values dx by which to update the state x.
    If there is exactly one vector dx for each vector x in the batch, the update is simply act_fn(x + dx).
    If this not the case, then either allocate a scalar dx to a scalar state, or allocate a scalar dx to a vector x at index idx -
    in this case the indices range over the length of the vector x.
    If the indices range over fixed transitions, no explicit dx is needed and idx collects the updates from a table of transitions.
    '''

    def __init__(self,
                 args: MutableMapping[str, Any],
                 config: MutableMapping[str, Any],
                 transition_table: Optional[TensorType] = None) -> None:
        super().__init__()
        
        self.args = args
        self.config = config

        self.batch_size = args.batch_size
        self.n_state_neurons = config.n_state_neurons
        self.n_actions = config.n_actions
        
        if transition_table is not None:
            assert transition_table.shape == (config.n_actions,)
            self.transition_table = transition_table

    def update(self,
               x: TensorType,
               state_transition: Optional[TensorType] = None,
               idx: Optional[pt.LongTensor] = None,
               act_fn: Optional[str] = 'id') -> Tuple[TensorType, TensorType]:
        
        with pt.no_grad():

            if state_transition is not None:
                if state_transition.shape == x.shape:
                    assert idx is None, 'state_transition and x have same shape, idx not necessary'
                    x = activation_fn(act_fn)(x + state_transition)
                elif state_transition.shape == (self.batch_size,):
                    if idx is None:
                        # state transition given, no indexing necessary
                        assert x.shape == (self.batch_size, 1)
                        x = activation_fn(act_fn)(x + state_transition.reshape(self.batch_size, 1))
                    elif idx is not None:
                        # the given state transition is 'allocated' at idx
                        assert idx.shape == (self.batch_size,)
                        x[range(self.batch_size), idx] += state_transition
                        x = activation_fn(act_fn)(x)

            elif state_transition is None:
                assert idx is not None and idx.shape == (self.batch_size,)
                state_transition = pt.gather(self.transition_table, 0, idx)
                state_transition = state_transition.view(x.shape)
                x = activation_fn(act_fn)(x + state_transition)

        return x, state_transition

    def clear(self) -> TensorType:
        x = pt.zeros(self.batch_size, self.n_state_neurons).to(self.args.device)
        return x