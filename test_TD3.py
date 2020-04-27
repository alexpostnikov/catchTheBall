
import os
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass

from os.path import dirname
cath_the_ball_dir = dirname(dirname(os.getcwd()))
sys.path.append(cath_the_ball_dir)
from environments.ur10_svh_env import ur10svh
from environments.ur10_svh_utils import load_yaml

from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import TD3

import numpy as np


cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))
# config = load_yaml(cath_the_ball_dir+"/configs/ur10_cfg.yaml")



if __name__ == "__main__":

    model_path = "best_model.pkl"

    env = ur10svh(cur_dir+"/ur10_cfg.yaml",
                    resource_directory=cath_the_ball_dir+"/rsc/ur10/", visualize=True)
    model = TD3.load(model_path)
    env = DummyVecEnv([env])



    obs = env.reset()

    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()