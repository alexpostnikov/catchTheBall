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

from multiprocessing import Process



def run_learning(ALGO, env_config_path, algo_config_path,video_folder):
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    env = ur10svh(cur_dir+env_config_path,
                    resource_directory="/media/robot/a8dd218a-4279-4bd4-b6af-8230c48776f7/iskander/schunk_gym/rsc/ur10/",video_folder=video_folder)  # gym.make('CartPole-v1')
    c_models = \
    {
        "PPO"  : c_PPO (algo_config_path, env, video_folder),
        "TRPO" : c_TRPO(algo_config_path, env, video_folder),
        "DDPG" : c_DDPG(algo_config_path, env, video_folder),
        "TD3"  : c_TD3 (algo_config_path, env, video_folder)
    }

    c_model = c_models[ALGO]()
    algo_config = load_yaml(algo_config_path)
    c_model.model.learn(total_timesteps=algo_config["total_timesteps"],tb_log_name="",callback=c_model.learning_callback) # 1 6000 000 ~1hr
    c_model.model.save(video_folder+"model.pkl")
    c_model.validate()
    


if __name__ == "__main__":

    algo_config_path ="./environments/ppo_cfg.yaml"
    env_config_path = "/environments/ur10_cfg.yaml"
    
    
    cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))    
    ALGO = "PPO"
    video_folder = check_video_folder(cur_dir+"/video/"+ALGO)
    video_folder = video_folder+"/"
    run_learning(ALGO,env_config_path,algo_config_path, video_folder)
    
    print ("done")
        