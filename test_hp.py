
import os
import sys
try:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
	pass

from environments.ur10_svh_env import ur10svh

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL
from stable_baselines.common import set_global_seeds, make_vec_env
from train_utils import *
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.td3.policies import MlpPolicy as tf3MlpPolicy
import argparse
from algos import c_DDPG, c_PPO, c_TD3, c_TRPO

import gym
from multiprocessing import Process
import shutil

algos = \
	{
		"PPO": PPO2,
		"TRPO": TRPO,
		"DDPG": DDPG,
		"TD3": TD3

	}


def yield_params():
	for lr in [1.0e-4, 2.5e-4, 5.0e-4]:
		for timesteps_per_batch in [1024, 512, 2048]:
			for gamma in [0.8, 0.9, 0.99]:
				for ent in [0.0, 0.01]:
					yield (lr, timesteps_per_batch, gamma, ent)


cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))


def run_learning(ALGO, env_config_path, algo_config_path, weight):
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	for lr, tpb, gamma, ent in yield_params():
		print ("starting: " + ALGO+"_lr_"+str(
			lr)+"_tpb_"+str(tpb) + "_g_" + str(gamma)+"_ent_"+str(ent))
		video_folder = check_video_folder(cur_dir+"/log/")
		video_folder = check_video_folder(cur_dir+"/log/"+ALGO+"_lr_"+str(
			lr)+"_tpb_"+str(tpb) + "_g_" + str(gamma)+"_ent_"+str(ent))
		video_folder = video_folder+"/"
		
		env = ur10svh(cur_dir+env_config_path,
					  resource_directory=cur_dir+"/rsc/ur10/", video_folder=video_folder)  # gym.make('CartPole-v1')
		c_models = \
			{
				"PPO": c_PPO(algo_config_path, env, video_folder),
				"TRPO": c_TRPO(algo_config_path, env, video_folder),
				"DDPG": c_DDPG(algo_config_path, env, video_folder),
				"TD3": c_TD3(algo_config_path, env, video_folder)
			}

		runner =\
			{
				"PPO":  cur_dir+"/test_ppo2.py",
				"TRPO": cur_dir+"/test_trpo.py",
				"DDPG": cur_dir+"/test_ddpg.py",
				"TD3":  cur_dir+"/test_TD3.py"
			}

		c_model = c_models[ALGO]()
		if (str(weight) != "None"):
			c_model.model = algos[ALGO].load(weight, c_model.env)
			c_model.model.tensorboard_log = video_folder

		### copy env and configs file to log folder ###
		shutil.copy2(algo_config_path, video_folder)
		shutil.copy2(cur_dir + env_config_path, video_folder)
		shutil.copy2(runner[ALGO], video_folder)
		shutil.copytree(cur_dir + "/environments/",
						video_folder+"/environments/")
		shutil.copytree(cur_dir + "/configs/", video_folder + "/configs/")

		c_model.gamma = gamma
		c_model.lr = lr
		c_model.n_steps = tpb
		c_model.ent_coef = ent

		c_model.learn()
		c_model.model.save(video_folder+"model.pkl")
		c_model.validate()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--jobs_config_path', type=str, default="./configs/jobs_cfg.yaml",
						help='path to config file')
	args = parser.parse_args()
	jobs_config = load_yaml(args.jobs_config_path)
	env_config_path = jobs_config["env_config_path"]
	cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))
	processes = []
	for i in range(0, jobs_config["num_jobs"]):
		ALGO = jobs_config["jobs"][i]["algo"]
		weight = jobs_config["jobs"][i]["weight"]
		algo_config_path = jobs_config["jobs"][i]["algo_config_path"]

		p = Process(target=run_learning, args=(
			ALGO, env_config_path, algo_config_path, weight))
		processes.append(p)
		p.start()

	for p in processes:
		p.join()

	print("done")
