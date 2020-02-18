#! /home/robot/venvs/raisimGymEnv/bin/python3.6

import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
from environments.ur10_svh_env import ur10svh

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL
from stable_baselines.common import set_global_seeds, make_vec_env
from train_utils import  *
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.td3.policies import  MlpPolicy as tf3MlpPolicy
import argparse
from algos import c_DDPG, c_PPO, c_TD3, c_TRPO
import tensorflow as tf
from train import run_learning
from multiprocessing import Process




if __name__ == "__main__":
    ALGO = "PPO"
    if ALGO == "TRPO":
        algo_config_path ="./environments/TRPO_cfg.yaml"
    if ALGO == "PPO":
        algo_config_path ="./environments/ppo_cfg.yaml"
    env_config_path = "/environments/ur10_cfg.yaml"
    
    cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))    
    
    video_folder = check_video_folder(cur_dir+"/video/"+ALGO)
    video_folder = video_folder+"/"
    run_learning(ALGO,env_config_path,algo_config_path, video_folder)

    print ("done")