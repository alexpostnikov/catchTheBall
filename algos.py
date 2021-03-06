from train_utils import *
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL
from stable_baselines.common import set_global_seeds, make_vec_env

from stable_baselines.ddpg.policies import LnMlpPolicy as ddpgLnMlpPolicy

from stable_baselines.td3.policies import MlpPolicy as tf3MlpPolicy
from stable_baselines.td3.policies import LnMlpPolicy as tf3LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
import tensorflow as tf
from datetime import datetime
import time

set_global_seeds(1)
import csv

class c_class():

    def __init__(self):
        self.video_folder = None
        self.model = None
        self.env = None
        self.algo = PPO2
        self.best_mean_reward = 0
        self.ep_infos = None
        self.timestamp = 0
        self.save_model_flag = True
        self.ep_infos = None
        self.saving_time = time.time()
        self.log_files = {}

    def __call__(self):
        raise NotImplementedError

    def validate(self):
        obs = self.env.reset()
        obs = self.env.reset()
        rew = []
        pose_rew = []
        ball_rew = []
        l = []
        r = []
        for i in range(5000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            if dones == True:
                rew.append(info["episode"]["r"])
                obs = self.env.reset()
                pose_rew.append(info["episode"]["pose_rew"])
                ball_rew.append(info["episode"]["ball_rew"])
                l.append(info["episode"]["l"])
                r.append(info["episode"]["r"])
                cur_step = info["episode"]["curriculum_step"]

        self.ep_infos = {}
        self.ep_infos["ball_rew"] = sum(ball_rew)/len(ball_rew)
        self.ep_infos["pose_rew"] = sum(pose_rew)/len(pose_rew)
        self.ep_infos["l"] = sum(l)/len(l)
        self.ep_infos["curriculum_step"] = cur_step
        self.ep_infos["r"] = sum(r)/len(r)

    def learn(self):
        self.model.learn(total_timesteps=self.algo_config["total_timesteps"], log_interval=100,
                         tb_log_name="", callback=self.learning_callback)  # 1 6000 000 ~1hr

    def move_next_curriculum(self):
        if isinstance(self.env, DummyVecEnv):
            for env in self.env.envs:
                env.update_cur = True
                env.ball_reward_buf.clear()
                env.pose_reward_buf.clear()
                env.pose_reward_averaged = 0.
                env.ball_reward_averaged = 0.
        else:
            self.env.ball_reward_buf.clear()
            self.env.pose_reward_buf.clear()
            self.env.pose_reward_averaged = 0.
            self.env.ball_reward_averaged = 0.
            self.env.update_cur = True

    def check_curriculum(self):
        quality = self.algo_config["quality"]
        if (isinstance(self.env, SubprocVecEnv)):
            return
        try:
            if isinstance(self.env, DummyVecEnv):
                do_upd = False
                for env in self.env.envs:
                    if ((env.pose_reward_averaged > quality) and (
                            env.ball_reward_averaged > quality))and (len(env.pose_reward_buf) >= env.buf_len):
                        self.move_next_curriculum()
                        return
            else:
                # if self.env.update_cur_inner:
                if (self.env.pose_reward_averaged > quality and self.env.ball_reward_averaged > quality)   \
                        and len(self.env.pose_reward_buf) >= self.env.buf_len:
                    self.move_next_curriculum()
        except ZeroDivisionError:
            print("zero division in check_curriculum!")

    @staticmethod
    def add_value(file, step,value):
        with open(file, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([time.time(), step, value])


    def learning_callback(self, locals, globals):
        
        self.check_curriculum()

        if locals["self"].num_timesteps - self.timestamp > self.algo_config["validate_every_timesteps"]:
            self.validate()
            self.timestamp = locals["self"].num_timesteps

        if self.save_model_flag == True:
            self.save_model_flag = False
            try:
                locals["self"].save(self.video_folder+"best_model.pkl")
            except:
                pass

        if time.time() - self.saving_time > 30*60:
            self.saving_time = time.time()
            locals["self"].save(self.video_folder + "model_" +
                                datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".pkl")
        if self.ep_infos is not None:
                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='episode_info/ep_len', simple_value=self.ep_infos["l"])])
                locals['writer'].add_summary(
                    summary, locals["self"].num_timesteps)

                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='episode_info/pose_rew', simple_value=self.ep_infos["pose_rew"])])
                locals['writer'].add_summary(summary, self.timestamp)
                self.add_value(self.log_files["pose_rew"], locals["self"].num_timesteps, self.ep_infos["pose_rew"])

                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='episode_info/ball_rew', simple_value=self.ep_infos["ball_rew"])])
                locals['writer'].add_summary(summary, self.timestamp)
                self.add_value(self.log_files["ball_rew"], locals["self"].num_timesteps, self.ep_infos["ball_rew"])

                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='episode_info/curriculum', simple_value=self.ep_infos["curriculum_step"])])
                locals['writer'].add_summary(summary, self.timestamp)
                self.add_value(self.log_files["curriculum"], locals["self"].num_timesteps, self.ep_infos["curriculum_step"])
                
                

                summary = tf.Summary(value=[tf.Summary.Value(
                    tag='episode_info/final_reward', simple_value=self.ep_infos["r"])])
                locals['writer'].add_summary(summary, self.timestamp)
                self.ep_infos = None
                


    def create_log_files(self):
        
        file_name = self.video_folder +"/ball_rew" + ".csv"
        self.log_files["ball_rew"] = file_name
        file_name = self.video_folder+"/curriculum" + ".csv"
        self.log_files["curriculum"] = file_name
        file_name = self.video_folder+"/pose_rew" + ".csv"
        self.log_files["pose_rew"] = file_name
        for _, file_name in self.log_files.items():
            print (file_name)
            with open(file_name, mode='a') as file_name:
                writer = csv.writer(file_name, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Wall time', 'Step', 'Value'])

class c_PPO(c_class):

    def __init__(self, config, env, video_folder):
        super(c_PPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        self.timestamp = 0
        self.save_model_flag = True
        self.ep_infos = None
        

    def set_algo_params(self):
        self.n_steps = self.algo_config["n_steps"]
        self.ent_coef = self.algo_config["ent_coef"]
        self.lr = self.algo_config["learning_rate"]
        self.gamma = self.algo_config["gamma"]
        self.policy_kwargs = dict(net_arch=[dict(pi=[self.algo_config["nn_size"],
                                                     self.algo_config["nn_size"]],
                                                 vf=[self.algo_config["nn_size"],
                                                     self.algo_config["nn_size"]])])

    def __call__(self):
        env_list = []
        print("self.algo_config[num_envs] ", self.algo_config["num_envs"])

        if self.algo_config["num_envs"] > 1:
            for _ in range(self.algo_config["num_envs"]):
                env_list.append(self.env)
            self.env = SubprocVecEnv(env_list, "fork")
        else:
            self.env = DummyVecEnv([self.env])
        self.algo = PPO2
        self.model = self.algo(MlpPolicy, self.env, verbose=self.algo_config["verbose"], tensorboard_log=self.video_folder,
                               n_steps=self.n_steps, ent_coef=self.ent_coef, learning_rate=self.lr,
                               gamma=self.gamma,
                               policy_kwargs=self.policy_kwargs)
        self.create_log_files()
        return self

    def validate(self):
        obs = self.env.reset()
        obs = self.env.reset()
        rew = []
        pose_rew = []
        ball_rew = []
        l = []
        r = []
        cur_step = 0
        for i in range(5000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)

            if isinstance(dones, np.ndarray):
                is_done = any(dones) == True
            else:
                is_done = dones
            if is_done:
                if not isinstance(dones, np.ndarray):
                    # if self.algo_config["num_envs"] == 1:
                    rew.append(info["episode"]["r"])
                    pose_rew.append(info["episode"]["pose_rew"])
                    ball_rew.append(info["episode"]["ball_rew"])
                    l.append(info["episode"]["l"])
                    cur_step = info["episode"]["curriculum_step"]
                    obs = self.env.reset()
                else:
                    # if self.algo_config["num_envs"] > 1:
                    info = info[np.where(dones == True)[0][0]]
                    rew.append(info["episode"]["r"])
                    pose_rew.append(info["episode"]["pose_rew"])
                    ball_rew.append(info["episode"]["ball_rew"])
                    l.append(info["episode"]["l"])
                    r.append(info["episode"]["r"])
                    cur_step = info["episode"]["curriculum_step"]

                    obs = self.env.reset()
        self.ep_infos = {}
        self.ep_infos["ball_rew"] = sum(ball_rew)/len(ball_rew)
        self.ep_infos["pose_rew"] = sum(pose_rew)/len(pose_rew)
        self.ep_infos["l"] = sum(l)/len(l)
        try:
            self.ep_infos["r"] = sum(r) / len(r)
        except ZeroDivisionError:
            self.ep_infos["r"] = 0
        self.ep_infos["curriculum_step"] = cur_step

        return rew


class c_TRPO(c_class):

    def __init__(self, config, env, video_folder):
        super(c_TRPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        self.algo = TRPO
        self.save_model_flag = True
        self.ep_infos = None
        self.timestamp = 0
        

    def set_algo_params(self):
        self.timesteps_per_batch = self.algo_config["timesteps_per_batch"]
        self.ent_coef = self.algo_config["ent_coef"]
        self.vf_stepsize = self.algo_config["learning_rate"]
        self.gamma = self.algo_config["gamma"]
        self.policy_kwargs = dict(net_arch=[dict(pi=[self.algo_config["nn_size"],
                                                     self.algo_config["nn_size"]],
                                                 vf=[self.algo_config["nn_size"],
                                                     self.algo_config["nn_size"]])])

    def __call__(self):
        self.model = self.algo(MlpPolicy, self.env,
                               tensorboard_log=self.video_folder, timesteps_per_batch=self.timesteps_per_batch,
                               gamma=self.gamma, vf_stepsize=self.vf_stepsize,
                               verbose=self.algo_config["verbose"], entcoeff=self.ent_coef,
                               policy_kwargs=dict(net_arch=[dict(pi=[self.algo_config["nn_size"], self.algo_config["nn_size"]],
                                                                 vf=[self.algo_config["nn_size"], self.algo_config["nn_size"]])]))
        self.create_log_files()
        return self

class c_DDPG(c_class):

    def __init__(self, config, env, video_folder):
        super(c_DDPG, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env

    def set_algo_params(self):
        self.policy = ddpgLnMlpPolicy
        self.tau = self.algo_config["tau"]
        self.batch_size = self.algo_config["timesteps_per_batch"]
        self.ent_coef = self.algo_config["ent_coef"]
        self.actor_lr = self.algo_config["actor_lr"]
        self.critic_lr = self.algo_config["critic_lr"]
        self.gamma = self.algo_config["gamma"]
        self.observation_range = (-1,1)
        self.return_range = (-1,1)

    def __call__(self):
        self.algo = DDPG
        # , verbose=1,random_exploration=0.02,return_range=(-1,1),observation_range=(-1,1))
        self.model = self.algo(self.policy, self.env, gamma=self.gamma,tau = self.tau, batch_size = self.batch_size,
                                actor_lr = self.actor_lr, critic_lr =self.critic_lr,  tensorboard_log=self.video_folder )
        self.create_log_files()
        return self


class c_TD3(c_class):
    def __init__(self, config, env, video_folder):
        super(c_TD3, self).__init__()
        self.algo_config = load_yaml(config)
        self.algo = TD3
        self.env = env
        self.video_folder = video_folder

    def set_algo_params(self):
        self.policy = tf3LnMlpPolicy
        self.tau = self.algo_config["tau"]
        self.batch_size = self.algo_config["timesteps_per_batch"]
        self.policy_noise = self.algo_config["policy_noise"] # 0.2
        self.random_exploration = self.algo_config["random_exploration"] #0.0
        self.lr = self.algo_config["learning_rate"] # 0.0003
        self.gamma = self.algo_config["gamma"]


    def __call__(self):
        self.model = self.algo(tf3MlpPolicy, self.env, gamma=self.gamma,learning_rate=self.lr,batch_size=self.batch_size,
                               tau=self.tau, target_policy_noise=self.policy_noise,
                               verbose=1, tensorboard_log=self.video_folder)
        self.create_log_files()
        return self
