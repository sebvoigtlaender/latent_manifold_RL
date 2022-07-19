# Reinforcement Learning on Latent Manifolds

PyTorch code for foundational experiments for learning state transition policies on latent manifolds.
The theoretical background for the experiments, drawing from systems and computational neuroscience as well as recent developments in artificial intelligence, is described in more detail [here](https://github.com/sebvoigtlaender/state_rl_basics/blob/main/background.pdf).

In short, we propose that one can reframe learning dynamical systems in latent space, conventionally done with a recurrent neural network, whose architecture and training algorithm depends on the specifics of the task, in the language of reinforcement learning. From this perspective an internal stage transition is understood as an action chosen by a policy acting on latent state variables, which is trained by using modified versions of conventional policy gradient algorithms.

The agent is tasked with learning the dynamics on a latent manifold, with specific constraints on the dynamics being imposed by using an optimization target either obtained by local search around the path in state space or by more sophisticated methods derived from variational free energy objectives.

In principle this allows performing temporal credit assignment by using value functions or more sophisticated methods for value transport without the need to do counterfactual reasoning in terms of partial derivatives along trajectories through state space, as conventionally done with backpropagation through time.

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
$ python main.py --batch-size=16
```

A run directory will be created in `runs/` containing TensorBoard metrics.