import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL

from train_utils import check_video_folder
from environments.ur10_svh_env import ur10svh
from stable_baselines.common.policies import MlpPolicy

from tensorboardX import SummaryWriter
import time

HP_timesteps_per_batch = hp.HParam('timesteps_per_batch', hp.Discrete([1024, 512, 248, 2048]))
HP_GAMMA = hp.HParam('gamma', hp.RealInterval(0.8,0.99))
HP_entcoeff = hp.HParam('entcoeff', hp.RealInterval(0.0, 0.05))

HP_timesteps_per_batch = [1024, 512, 248, 2048]
HP_GAMMA = [0.8,0.9,0.99]
HP_entcoeff = [0.0,0.01]


METRIC_ACCURACY = 'accuracy'


session_num = 0

with SummaryWriter("Hp_log_big_TRPO") as w:
	for timesteps_per_batch in HP_timesteps_per_batch:
		for GAMMA in HP_GAMMA:
			for entcoeff in HP_entcoeff:
				name = "t" + str(timesteps_per_batch) + "G"+ str(GAMMA)[0:5] + "e" + str(entcoeff)[0:4]
				step = 0 
				for i in range(50):
					start_time = time.time()

					if step == 0:
						cur_dir = os.path.dirname(os.path.abspath(__file__))
						video_folder = check_video_folder(cur_dir+"/video/"+"TRPO")
						video_folder = video_folder+"/"
						env = ur10svh(cur_dir+"/environments/ur10_cfg.yaml",
									resource_directory="/media/robot/a8dd218a-4279-4bd4-b6af-8230c48776f7/iskander/schunk_gym/rsc/ur10/",video_folder=video_folder)  # gym.make('CartPole-v1')

						model = TRPO(MlpPolicy, env, gamma=GAMMA,timesteps_per_batch=timesteps_per_batch, entcoeff=entcoeff, verbose=0, tensorboard_log=video_folder)
						
					model.learn(total_timesteps=60000, tb_log_name="") # 1 6000 000 ~1hr
					model.save(video_folder+"session_num"+str(session_num)+"_step_"+str(step)+".pkl")
					session_num += 1
					
					# VALIDATION
					obs = env.reset()
					obs = env.reset()
					rew = []
					
					for i in range(1000):
						action, _states = model.predict(obs)
						obs, rewards, dones, info = env.step(action)
						if dones == True:
							rew.append(info["episode"]["l"])
							obs = env.reset()
					rew = sum(rew)/len(rew)
					hparams = {
						"HP_timesteps_per_batch": timesteps_per_batch,
						"HP_GAMMA": GAMMA,
						"HP_entcoeff": entcoeff,
					}
					step += 60000
					w.add_hparams(hparams, {'hparam/av_len': rew},name = name, global_step=step)
					
					print (hparams)
					print (rew)
					print ("done in ", time.time() - start_time)
					print ("-----------------------------")

				
		