# Reinforcement Learning on Latent Manifolds

PyTorch code for basic functionalities. Theoretical background described [here](https://github.com/sebvoigtlaender/state_rl_basics/blob/main/log.pdf)

## Package Description

`arguments.py` contains all available hyperparameters.

`env.py` contains the environment configuration as well as three test environments:
*   a toggle environment with a binary input signal `u` and a binary output signal `x_ground_truth`
*   a maze environment without walls
*   a maze environment with randomly generated walls

`helpers.py` contains two helper functions while calculate a locally optimal transition and a locally optimal probability distribution target.

`main.py` is responsible for setting up the environment, the trainer, the model, and the optimizer as well as executing the code. Results can be looked up in the directory `runs/` containing TensorBoard metrics.

`MLE_x.py` and `MLE_dx.py` contains runnable code for a REINFORCE-like and basic Actor-Critic-like algorithms.

`model.py` contains the model configuration as well as code for the core model, the actor critic, and the state update reservoir.

`optimizer.py` contains code for configurable basic optimizer or policy gradient algorithms, and one optimize for evolutionary approaches and swarm optimization.

`trainer.py` contains the trainer class code.

`utils.py` contains code for basic utility functions such as custom typing, customizable activation and loss functions, as well as some further helper functions.

### Training

Basic usage is

```bash
$ python main.py
```

See [arguments.py](https://github.com/sebvoigtlaender/state_rl_basics/blob/main/arguments.py) for available parameters, and set parameters using e.g.:

```bash
$ python main.py -batch-size=16
```

A run directory will be created in `runs/` containing TensorBoard metrics.

### Tests

Unit tests will follow.

### Lessons learned