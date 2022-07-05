# Reinforcement Learning on Latent Manifolds

PyTorch code for basic functionalities. Theoretical background described [here](state_rl_basics/blob/main/log.pdf)

## Package Description

Testing on `HalfCheetah-v2` and `Walker2d-v2`:

![](images/results.png)

### Training

Basic usage is

```bash
$ python main.py
```

See [arguments.py](state_rl_basics/arguments.py) for available parameters, and set parameters using e.g.:

```bash
$ python main.py -batch-size=16
```

A run directory will be created in `runs/` containing TensorBoard metrics.

### Tests

To run unit tests, respectively:

Unit tests cover:
*   


## Lessons learned

### Stochastic policies