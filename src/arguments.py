import argparse
import imp
from re import T
from tkinter.tix import Tree
from tokenize import Triple
from args_inverted_double_pendulum import get_inverted_double_pendulum_args
from args_hopper import get_hopper_args
from args_walker2d import get_walker2d_args
from args_halfcheetah import get_halfcheetah_args
from args_ant import get_ant_args
from args_humanoid import get_humanoid_args


env_name = 'Walker2d-v2'
# 'Hopper-v2' 'Walker2d-v2' 'InvertedDoublePendulum-v2' 'HalfCheetah-v2' 'Ant-v2' 'Humanoid-v2'

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--cuda_num', default='1')
    parser.add_argument('--use_wandb', default=True)
    parser.add_argument('--user_name', default='') # need to specify
    parser.add_argument('--n_training_threads', default=23)
    parser.add_argument('--experiment_name', default='S1')
    parser.add_argument('--train_model_by_RL', default=True)
    parser.add_argument('--MPCmodel', default=True)
    parser.add_argument('--max_plan_times', default=10)

    if env_name == 'Hopper-v2':
        get_hopper_args(parser)
    elif env_name == 'Walker2d-v2':
        get_walker2d_args(parser)
    elif env_name == 'HalfCheetah-v2':
        get_halfcheetah_args(parser)
    elif env_name == 'InvertedDoublePendulum-v2':
        get_inverted_double_pendulum_args(parser)
    elif env_name == 'Ant-v2':
        get_ant_args(parser)
    elif env_name == 'Humanoid-v2':
        get_humanoid_args(parser)

    return parser.parse_args()
