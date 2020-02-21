#! /home/robot/venvs/raisimGymEnv/bin/python3.6
import os
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass
import gym

from environments.ur10_svh_env import ur10svh
from environments.ur10_svh_utils import load_yaml
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL
from stable_baselines.common import set_global_seeds, make_vec_env

import argparse
import numpy as np


best_mean_reward = -np.inf
rsg_root = os.path.dirname(os.path.abspath(__file__)) + ''
log_dir = rsg_root+"/logs/"
ALGO = "PPO"
cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))
config = load_yaml(cur_dir+"/environments/ur10_cfg.yaml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default="/home/robot/repos/catchTheBall/log/best_model.pkl",
                    help='algo type, extect one of: PPO, TRPO, DDPG or TD3')
    

    args = parser.parse_args()
    model_path = args.model_path

    env = ur10svh(cur_dir+"/environments/ur10_cfg.yaml",
                    resource_directory=cur_dir+"/rsc/ur10/", video_folder="./video/TRPO/")  # gym.make('CartPole-v1')

    #model_path = "/media/robot/a8dd218a-4279-4bd4-b6af-8230c48776f7/stableBaselines/ur10_svh/video/PPO_4/best_model.pkl"
    model = TRPO.load(model_path)

    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor

    env = DummyVecEnv([env])
    # env = VecNormalize(env)



    obs = env.reset()

    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()