# environments and signal generators
import argparse
from typing import Any, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import numpy.random as rnd
import torch as pt
from absl import logging

import gym
from gym import spaces
from gym.utils import seeding

from utils import __add__, ListType, TensorDict, TensorType 


logging.set_verbosity(logging.INFO)


class ALE():

    def __init__(self):
        self.lives = lambda: 0

# self.codes
# self.n_codes = len(self.codes)
# self.n_codes = 5
# n_input_neurons = self.n_codes + 1
# barcodes = np.stack([rnd.choice([-1, 1], self.d_cue) for i in range(self.n_codes)])


def get_env_config(args: MutableMapping[str, Any]) -> Mapping[str, Any]:

    '''set up immutable environment configuration'''
    
    parser = argparse.ArgumentParser()
    config = parser.parse_args(args=[])

    config.batch_order = 'time_first'
    config.batch_size = args.batch_size
    config.device = args.device

    if not hasattr(args, 'env_id'):
        logging.fatal('env_id must be specified')
    else:
        logging.info(f'environment = {args.env_id}')
        config.env_id = args.env_id

    if args.env_id == 'toggle':
        config.d_cue = args.d_cue
        config.len_cue_sequence = args.len_cue_sequence
        config.n_cues = args.n_cues

        config.n_actions = 3

    elif args.env_id == 'square_full_access':
        config.height = args.height
        config.n_steps = args.n_steps

        config.initial_idx = (0,)
        config.n_actions = config.height**2

    elif args.env_id == 'square_with_walls':
        config.height = args.height
        config.n_steps = args.n_steps

        config.initial_idx = (0, 0)
        config.n_actions = 4

    args.n_actions = config.n_actions

    return config


def get_env(config: Mapping[str, Any]) -> Any:
    if config.env_id == 'toggle':
        return Toggle(config)
    elif config.env_id == 'square_full_access':
        return Maze(config)
    elif config.env_id == 'square_with_walls':
        return Maze(config)

        
class Toggle(gym.Env):
    '''
    Toggle environment for basic internal state control tasks, inherits from gym environment
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Mapping[str, Any]) -> None:

        super().__init__()
        self.config = config
        if config.env_id == 'toggle':
            print(config.env_id)
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(config.d_cue,))
            self.len_cue_sequence = config.len_cue_sequence
            self.d_cue = config.d_cue
            self.n_cues = config.n_cues
            self.batch_size = config.batch_size

            self.seed()

            # Needed by atari_wrappers in OpenAI baselines
            self.ale = ALE()
            seed = None
            self.np_random, np_seed = seeding.np_random(seed)

    def get_action_meanings(self) -> None:
        return ['DECREASE STATE', 'NO UPDATE', 'INCREASE STATE']

    def _generate_cue(self) -> None:

        if self.config.env_id == 'toggle':
            assert self.d_cue == 1
            idcs = [np.sort(rnd.choice(range(1, self.len_cue_sequence-1), self.n_cues, replace=0)) for i in range(self.batch_size)]
            idcs = np.stack(idcs)
            self.cue_sequence = np.zeros((self.batch_size, self.len_cue_sequence))
            self.target = np.zeros((self.batch_size, self.len_cue_sequence))
            for i in range(self.batch_size):
                self.cue_sequence[i, idcs[i]] = 1.0
                for (t0, t1) in list(zip(np.append(0, idcs[i]), np.append(idcs[i], self.len_cue_sequence)))[1::2]:
                    self.target[i, t0:t1] = 1.0

        if self.config.env_id == '':
            code_idx = rnd.choice(range(self.n_codes), (self.batch_size, self.n_cues))
            spike_idx = np.zeros((self.batch_size, self.n_cues), dtype=int)
            for i in range(self.batch_size):
                spike_idx[i] = np.sort(rnd.choice(range(self.len_cue_sequence), self.n_cues, replace=0))

            # self.cue_sequence = np.zeros((self.batch_size, self.len_cue_sequence, self.n_codes))
            # for t in range(self.n_cues):
            #     self.cue_sequence[range(self.batch_size), spike_idx[range(self.batch_size), t], code_idx[range(self.batch_size), t]] = 1
            # self.target = np.cumsum(self.cue_sequence, 1)
            # self.target = np.sum(self.cue_sequence, 1)

            self.cue_sequence = np.zeros((self.batch_size, self.len_cue_sequence, self.d_cue))
            self.target = np.zeros((self.batch_size, self.len_cue_sequence, self.n_codes))
            for t in range(self.n_cues):
                self.cue_sequence[range(self.batch_size), spike_idx[range(self.batch_size), t]] = self.codes[code_idx[range(self.batch_size), t]]
                self.target[range(self.batch_size), spike_idx[range(self.batch_size), t], code_idx[range(self.batch_size), t]] = 1
            self.target = np.cumsum(self.target, 1)
        #     self.target = np.sum(self.target, 1)

        assert hasattr(self.config, 'batch_order'), 'specify batch order as config.batch_order'
        if self.config.batch_order == 'time_first':
            self.cue_sequence = self.cue_sequence.T.reshape((self.len_cue_sequence, self.batch_size, self.d_cue))
            self.target = self.target.T.reshape((self.len_cue_sequence, self.batch_size, self.d_cue))
        elif self.config.batch_order == 'batch_first':
            self.cue_sequence = self.cue_sequence.reshape((self.batch_size, self.len_cue_sequence, self.d_cue))
            self.target = self.target.reshape((self.batch_size, self.len_cue_sequence, self.d_cue))

    def seed(self, seed: Optional[bool] = None) -> List[Any]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[TensorType, TensorType, bool]:
        self.steps = 0
        self._generate_cue()
        x, state_target = self.cue_sequence[self.steps], self.target[self.steps]
        self.steps += 1
        return x, state_target, False

    def step(self, action: Optional[Union[List[int], TensorType]] = None) -> Tuple[ListType, ListType, bool]:
        assert self.steps > 0
        x, state_target = self.cue_sequence[self.steps], self.target[self.steps]
        self.steps += 1
        done = False
        if self.steps == self.len_cue_sequence:
            done = True
        return x, state_target, done

    def render(self, mode: str = 'human', close: Optional[bool] = False) -> Tuple[ListType, ListType, int]:
        return self.cue_sequence, self.target, self.steps


class Maze(gym.Env):
    '''
    Maze for basic spatial navigation tasks, inherits from gym environment
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Mapping[str, Any], transition_table: List = None) -> None:

        super().__init__()
        self.config = config
        self.n_steps = config.n_steps

        if config.env_id == 'square_full_access':
            assert config.n_actions == config.height**2, ''
            print(config.env_id)
            self.action_space = spaces.Discrete(config.n_actions)
            self.observation_space = spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(config.n_actions,))
            self.board = self._generate_maze()
            self.current_state = None
            assert len(config.initial_idx) == 1
            self.initial_idx = config.initial_idx # (0,)

            self.goal_state = np.copy(self.board)
            self.goal_state[config.n_actions - 1] = 1.0

        elif config.env_id == 'square_with_walls':
            if transition_table is None:
                self.transition_table = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            else:
                self.transition_table = transition_table
            print(config.env_id)
            assert config.n_actions == len(self.transition_table), 'n_actions must match n_transitions'
            self.action_space = spaces.Discrete(config.n_actions)
            self.observation_space = spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(config.height, config.height))
            self.board = self._generate_maze()
            self.current_state = None
            try:
                config.initial_idx
            except:
                config.initial_idx = (0, 0)
            
            assert len(config.initial_idx) == 2
            self.initial_idx = config.initial_idx

            self.goal_state = np.zeros((config.height, config.height))
            self.goal_state[(config.height-1, config.height-1)] = 1.0
            self.goal_state = self.goal_state.reshape((config.height**2))

        self.seed()

        # Needed by atari_wrappers in OpenAI baselines
        self.ale = ALE()
        seed = None
        self.np_random, np_seed = seeding.np_random(seed)

    def get_action_meanings(self) -> None:
        if self.config.env_id == 'square_full_access':
            return ['GO TO FIELD <ACTION>']
        elif self.config.env_id == 'square_with_walls':
            if self.config.n_actions == 4:
                return ['UP', 'RIGHT', 'DOWN', 'LEFT']
            if self.config.n_actions == 5:
                return ['NOOP', 'UP', 'RIGHT', 'DOWN', 'LEFT']

    def _generate_maze(self) -> ListType:
        if self.config.env_id == 'square_full_access':
            self.board = np.zeros(self.config.n_actions)
        elif self.config.env_id == 'square_with_walls':
            self.board = np.zeros((self.config.height+2, self.config.height+2))
            self.board[1:self.config.height+1, 1:self.config.height+1] = 1
            #self.board[1:self.config.height+1, 1:self.config.height+1] = rnd.choice([0, 1], (self.config.height, self.config.height), p = (.2, .8))
        return self.board

    def seed(self, seed: Optional[bool] = None) -> List[Any]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[ListType, bool]:
        self.steps = 0
        if self.config.env_id == 'square_full_access':
            self.current_state = np.copy(self.board)
            self.current_position = self.initial_idx
            self.current_state[self.current_position] = 1.0
            state = self.current_state

        if self.config.env_id == 'square_with_walls':
            self.current_state = np.zeros((self.config.height, self.config.height))
            self.current_position = self.initial_idx
            self.current_state[self.current_position] = 1.0
            state = self.current_state.reshape(1, self.config.height**2)

        return state, False

    def step(self, action: Union[int, TensorType]) -> Tuple[ListType, ListType, bool]:
        assert self.steps >= 0

        if self.config.env_id == 'square_full_access':
            self.current_state = np.copy(self.board)
            self.current_state[action] = 1
            state = self.current_state

        if self.config.env_id == 'square_with_walls':
            self.current_state[self.current_position] -= 1.0
            dx = self.transition_table[action]
            if self.board[__add__(__add__(self.current_position, (1, 1)), dx)]:
                self.current_position = __add__(self.current_position, dx)
            self.current_state[self.current_position] += 1.0
            state = self.current_state.reshape(1, self.config.height**2)
        
        self.steps += 1
        reward = 0
        done = False
        if np.all(state == self.goal_state):
            reward = 1
            done = True
        elif self.steps == self.n_steps:
            reward = 0
            done = True
        return state, reward, done

    def render(self, mode: str = 'human', close: Optional[bool] = False) -> ListType:
        if self.current_state is not None:
            state = self.current_state.reshape(self.config.height, self.config.height)
            return state
        else:
            raise ValueError('no current state, cannot be rendered')