
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
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

cur_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    model_path = "best_model.pkl"

    env = ur10svh(cur_dir + "/ur10_cfg.yaml",
                  resource_directory=cath_the_ball_dir + "/rsc/ur10/", visualize=True, video_folder="./video/")
    env = DummyVecEnv([env])
    model = DDPG.load(model_path)
    obs = env.reset()

    for i in range(10000):

        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if (dones == True):

            env.envs[0].vis.stop_recording_video_and_save()
            env.envs[0].reset()
            env.envs[0].in_recording = False
        env.render()
    env.close()