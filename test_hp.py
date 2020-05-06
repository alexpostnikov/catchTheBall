
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

import threading


import multiprocessing


import concurrent.futures




algos = \
	{
		"PPO": PPO2,
		"TRPO": TRPO,
		"DDPG": DDPG,
		"TD3": TD3

	}


def yield_params():
	for ac_lr in [1.0e-4, 2.5e-4]:
		for crit_lr in [1.0e-3, 2.5e-3]:
			for timesteps_per_batch in [256, 512]:
				for gamma in [0.8,  0.99]:
					for tau in [0.002, 0.001]:
						yield (ac_lr, crit_lr, timesteps_per_batch, gamma, tau)

cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))




def learning_process(model, video_folder):

	model.learn()
	model.model.save(video_folder+"model.pkl")
	model.validate()	



def run_learning(ALGO, env_config_path, algo_config_path, weight, ac_lr, cr_lr,tpb, gamma, tau):
			cur_dir = os.path.dirname(os.path.abspath(__file__))
			print ("starting: " + ALGO+"_lr_"+str(
				ac_lr)+"_tpb_"+str(tpb) + "_g_" + str(gamma)+"_ent_"+str(tau))
			video_folder = check_video_folder(cur_dir+"/log/", False)
			video_folder = check_video_folder(cur_dir+"/log/"+ALGO+"_ac_lr_"+str(
				ac_lr)+"_cr_lr_"+str(cr_lr)+"_tpb_"+str(tpb) + "_g_" + str(gamma)+"_ent_"+str(tau))
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

			c_models[ALGO].set_algo_params()
			c_models[ALGO].gamma = gamma
			c_models[ALGO].critic_lr = cr_lr
			c_models[ALGO].actor_lr = ac_lr
			c_models[ALGO].batch_size = tpb
			c_models[ALGO].tau = tau
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
			# learning_process(c_model,video_folder)
			c_model.learn()
			c_model.model.save(video_folder+"model.pkl")
			c_model.validate()	
			# learning_process(c_model, video_folder)


if __name__ == "__main__":
	multiprocessing.set_start_method('forkserver', True)
	jobs_config_path = "./configs/jobs_cfg.yaml"

	jobs_config = load_yaml(jobs_config_path)
	env_config_path = jobs_config["env_config_path"]
	cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))
	ALGO = jobs_config["jobs"][0]["algo"]
	weight = jobs_config["jobs"][0]["weight"]
	algo_config_path = jobs_config["jobs"][0]["algo_config_path"]
	with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
		for ac_lr, cr_lr,tpb, gamma, tau in yield_params():
			executor.submit (run_learning,ALGO, env_config_path, algo_config_path, weight, ac_lr, cr_lr,tpb, gamma, tau)
	

	print("done")
