import argparse
import torch as pt

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--batch-size', type=int, default=16, help='batch size (default: 16)')
    parser.add_argument(
        '--d-cue', type=int, default=1, help='cue dimension (default: 1)')
    parser.add_argument(
        '--device', default='cpu', help='device (default: cpu)')
    parser.add_argument(
        '--discount', default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument(
        '--exploration-noise', default=1.0, help='standard deviation of exploration noise (default: 1.0)')
    parser.add_argument(
        '--lr', type=float, default=3e-3, help='learning rate (default: 3e-3)')
    parser.add_argument(
        '--n-episodes', type=int, default=int(1e4), help='number of training episodes (default: 1e4)')
    parser.add_argument(
        '--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--std', default=1e-1, help='standard deviation for evolution strategy (default: 1e-1)')
    parser.add_argument(
        '--tau', default=0.005, help='gradient decay rate for target networks (default: 0.005)')

    args = parser.parse_args(args=[])
    args.cuda = not args.no_cuda and pt.cuda.is_available()

    return args