import argparse
from typing import Any, Mapping, MutableMapping, Optional

import torch as pt

from absl import logging

from arguments import get_args
from env import get_env_config, get_env, Toggle, Maze
from model import get_actor_critic_config
from trainer import Trainer
from utils import fix_seed, TensorDict, TensorType 

logging.set_verbosity(logging.INFO)

args = get_args()
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
'''Specify environment, loss function, sampling strategy'''
args.device = device
args.env_id = 'toggle'
args.loss_fn = 'kl_div'
args.seed = False
args.stochastic = False

def main(args: MutableMapping[str, Any]) -> None:

    fix_seed(args)
    env_config = get_env_config(args)
    actor_critic_config = get_actor_critic_config(args)

    logging.info(f'hyperp: {args}')
    logging.info(f'env_config: {env_config}')
    logging.info(f'actor_critic_config: {actor_critic_config}')

    env = get_env(env_config)
    trainer = Trainer(args, actor_critic_config)
    
    trainer.train(env)

if __name__ == "__main__":
    main(args)