
import os
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass

sys.path.append("/home/robot/repos/catchTheBall")
from environments.ur10_svh_env import ur10svh
from environments.ur10_svh_utils import load_yaml

from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import TRPO

import numpy as np


best_mean_reward = -np.inf
rsg_root = os.path.dirname(os.path.abspath(__file__)) + ''
log_dir = rsg_root+"/logs/"
cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))
config = load_yaml("/home/robot/repos/catchTheBall/"+"/environments/ur10_cfg.yaml")

if __name__ == "__main__":

    model_path = "best_model.pkl"
    env = ur10svh("/home/robot/repos/catchTheBall/environments/ur10_cfg.yaml",
                    resource_directory="/home/robot/repos/catchTheBall/rsc/ur10/", visualize=True)

    model = TRPO.load(model_path)
    env = DummyVecEnv([env])



    obs = env.reset()

    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()