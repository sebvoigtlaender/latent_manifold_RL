import argparse, os
from tqdm import tqdm
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Union

import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging
from torch.utils.tensorboard import SummaryWriter

from arguments import get_args
from env import get_env_config, get_env, Toggle, Maze
from helpers import get_p_action_target, value_act
from model import get_actor_critic_config, ActorCritic, Reservoir
from optimizer import CoreOptimizer
from utils import activation_fn, convert_to_tensor, fix_seed, join_tokens, loss_fn, TensorDict, TensorType 

logging.set_verbosity(logging.INFO)


class Trainer():

    '''
    Responsible for setting up model (including embedders and task modules), optimizer etc., as well as training and evaluation
    '''

    def __init__(self,
                 args: MutableMapping[str, Any],
                 actor_critic_config: Mapping[str, Any], 
                 output_dir: Optional[str] = None) -> None:
        
        self.save_iteration = args.save_iteration
        self.log_iteration = args.log_iteration
        self.output_dir = output_dir
        self.writer = SummaryWriter()

        self.transition_table = pt.Tensor([-1, 0, 1])
        assert args.n_actions == len(self.transition_table)

        self.actor_critic = ActorCritic(actor_critic_config).to(args.device)
        self.actor_target = ActorCritic(actor_critic_config).to(args.device)
        self.actor_target.load_state_dict(self.actor_critic.state_dict(), strict=False)
        self.loss_fn = loss_fn(args.loss_fn)
        self.reservoir = Reservoir(actor_critic_config, self.transition_table).to(args.device)

        if args.optimizer_class == 'adam':
            self.optimizer = pt.optim.Adam(self.actor_critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)       
        elif args.optimizer_class == 'rms_prop':
            self.optimizer = pt.optim.RMSprop(self.actor_critic.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer_class == 'core':
            self.optimizer = CoreOptimizer(self.actor_critic.parameters(), lr=args.lr, args=args)
        elif args.optimizer_class == 'swarm':
            self.optimizer = SwarmOptimizer(self.actor_critic.parameters(), lr=args.lr, args=args)
        else:
            raise NotImplementedError()

        self.global_step = 1
        if hasattr(args, 'scheduler'):
            self.schedulers = utils.scheduler
        if hasattr(args, 'timer'):
            self.timer = utils.timer

        self.state_dict_path = f'state_dict/actor_critic_{args.env_id}'
        if self.output_dir:
            if os.path.isdir(self.output_dir):
                logging.info(f'Output directory "{self.output_dir}" already exists. Calling train will attempt to start from last checkpoint.')
            os.makedirs(self.output_dir, exist_ok=True)

        self.args = args

    def _save_states(self):

        '''Saves current step, model parameters, and optimizer parameters'''
        
        state = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': int(self.global_step)
        }
        pt.save(state, self.state_dict_path)
        
    def _load_states(self,
                     state_dict_path: str,
                     step: Optional[int] = None) -> None:

        '''Loads current step, model parameters, and optimizer parameters'''
        
        if not os.path.exists(state_dict_path):
            raise ValueError(f'No file `{state_dict_path}`.')

        state = pt.load(state_dict_path)
        step = state.get['global_step']
        self.actor_critic.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.global_step.fill_(step)
        logging.info(f'Loaded state from step {step}.')
        return step

    def train(self,
              env: Union[Iterator, Any],
              load: Optional[bool] = False,
              state_dict_path: Optional[str] = None) -> None:

        if load:
            start_step = self._load_states(state_dict_path)
        else:
            start_step = self.global_step
        logging.info('start training')
        self.actor_critic.train(True)

        u, state_target, done = env.reset()
        if hasattr(self, 'reservoir'):
            x = self.reservoir.clear()

        p_action_list = []
        p_action_target_list = []

        for t in tqdm(range(start_step, self.args.n_episodes + start_step)):

            u, state_target = convert_to_tensor(u), convert_to_tensor(state_target)
            p_action, action = self.actor_critic.act(join_tokens(u, x))
            p_action_target = get_p_action_target(self.args, x, state_target, self.transition_table)
            x, state_transition = self.reservoir.update(x, idx = action)
            p_action_list.append(p_action.probs)
            p_action_target_list.append(p_action_target)

            u, state_target, done = env.step()
            self.global_step += 1

            if t % self.save_iteration == 0:
                self._save_states()

            if done:
                p_action_list = pt.stack(p_action_list, 0)
                p_action_target_list = pt.stack(p_action_target_list, 0)
                loss = self.loss_fn(pt.log(p_action_list), p_action_target_list, reduction='batchmean')
                self.writer.add_scalar('Loss/train', loss.item(), t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                p_action_list = []
                p_action_target_list = []
                u, state_target, done = env.reset()
                x = self.reservoir.clear()

        self.writer.close()