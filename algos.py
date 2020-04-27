from train_utils import  *
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL
from stable_baselines.common import set_global_seeds, make_vec_env

from environments.multy_tread_env import RaisimGymVecEnv
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy as ddpgLnMlpPolicy

from stable_baselines.td3.policies import  MlpPolicy as tf3MlpPolicy
from stable_baselines.td3.policies import  LnMlpPolicy as tf3LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
import tensorflow as tf
from datetime import datetime
import time

set_global_seeds(1)
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
        rew = sum(rew)/len(rew)

        rew = rew * (cur_step + 1) ### next curriculum -> better model apriory?
        self.ep_infos = {}
        self.ep_infos["ball_rew"] = sum(ball_rew)/len(ball_rew)
        self.ep_infos["pose_rew"] = sum(pose_rew)/len(pose_rew)
        self.ep_infos["l"] = sum(l)/len(l)
        self.ep_infos["curriculum_step"] = cur_step
        self.ep_infos["r"] = sum(r)/len(r)

        if rew > self.best_mean_reward:
                self.best_mean_reward = rew
                self.save_model_flag = True
        return rew

    def learn(self):

        self.model.learn(total_timesteps=self.algo_config["total_timesteps"], log_interval=1000, tb_log_name="", callback=self.learning_callback) # 1 6000 000 ~1hr

    def check_curriculum(self):
        quality = self.algo_config["quality"]
        if (isinstance(self.env, SubprocVecEnv)):
            return
        try:
            if isinstance(self.env, DummyVecEnv):
                do_upd = False
                for env in self.env.envs:
                    if ((env.pose_reward_averaged > quality) and (
                            env.ball_reward_averaged > quality))and (len(env.pose_reward_buf) >= env.buf_len ):
                        do_upd = True
                if do_upd:
                    for env in self.env.envs:
                        env.update_cur = True
                # for env in self.env.envs:
                        env.ball_reward_buf.clear()
                        env.pose_reward_buf.clear()
                        env.pose_reward_averaged = 0.
                        env.ball_reward_averaged = 0.
            else:
                # if self.env.update_cur_inner:
                    if (self.env.pose_reward_averaged > quality and self.env.ball_reward_averaged > quality)   \
                            and len(self.env.pose_reward_buf) >= self.env.buf_len:
                        self.env.update_cur = True
                        self.env.ball_reward_buf.clear()
                        self.env.pose_reward_buf.clear()
                        self.env.pose_reward_averaged = 0.
                        self.env.ball_reward_averaged = 0.

        except ZeroDivisionError:
            print("zero division in check_curriculum!")

    def learning_callback(self,locals, globals):

        # locals["self"].vf_stepsize = np.clip(np.random.normal(0.0003, 0.0003), 0.0001,0.001)
        self.check_curriculum()

        if  locals["self"].num_timesteps - self.timestamp > self.algo_config["validate_every_timesteps"]:
            self.validate()
            self.timestamp = locals["self"].num_timesteps

        if self.save_model_flag == True:
            self.save_model_flag = False
            try:
                locals["self"].save(self.video_folder+"best_model.pkl")
            except:
                pass

        if time.time() - self.saving_time > 600:
            self.saving_time = time.time()
            locals["self"].save(self.video_folder + "model_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".pkl")
        try:
            if self.ep_infos is not None:
                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/ep_len', simple_value = self.ep_infos["l"] )])
                locals['writer'].add_summary(summary, locals["self"].num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/pose_rew', simple_value=self.ep_infos["pose_rew"] )])
                locals['writer'].add_summary(summary, self.timestamp)
                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/ball_rew', simple_value=self.ep_infos["ball_rew"] )])
                locals['writer'].add_summary(summary, self.timestamp)
                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/curriculum', simple_value=self.ep_infos["curriculum_step"] )])
                locals['writer'].add_summary(summary, self.timestamp)
                summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/final_reward', simple_value=self.ep_infos["r"] )])
                locals['writer'].add_summary(summary, self.timestamp)
        except:
            pass

class c_PPO(c_class):

    def __init__(self,config, env, video_folder):
        super(c_PPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        self.timestamp = 0
        self.save_model_flag = True
        self.ep_infos = None

    def __call__(self):
        env_list = []
        print("self.algo_config[num_envs] ",self.algo_config["num_envs"])
        # self.env = RaisimGymVecEnv(self.env, 10, 10)
        if self.algo_config["num_envs"] > 1:
            for _ in range(self.algo_config["num_envs"]):
                env_list.append(self.env)
            # self.env = DummyVecEnv(env_list)
            self.env = SubprocVecEnv(env_list, "fork")
        else:
            self.env = self.env

        self.algo = PPO2
        # self.model = self.algo(MlpLnLstmPolicy, self.env, n_steps=self.algo_config["n_steps"], nminibatches=1,
        #                         verbose=self.algo_config["verbose"], ent_coef=self.algo_config["ent_coef"],learning_rate=self.algo_config["learning_rate"],
        #                         gamma=self.algo_config["gamma"], tensorboard_log=self.video_folder)
        self.model = self.algo(MlpPolicy, self.env, n_steps=self.algo_config["n_steps"],
                                verbose=self.algo_config["verbose"], ent_coef=self.algo_config["ent_coef"],learning_rate=self.algo_config["learning_rate"],
                                gamma=self.algo_config["gamma"], tensorboard_log=self.video_folder,
                                policy_kwargs=dict(net_arch=[dict(pi=[self.algo_config["nn_size"],
                                    self.algo_config["nn_size"]], vf=[self.algo_config["nn_size"], self.algo_config["nn_size"]])]))
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
            action, _states = self.model.predict([obs])
            obs, rewards, dones, info = self.env.step(action[0])

            if isinstance(dones, np.ndarray):
            # if self.algo_config["num_envs"]> 1:
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

        # rew = rew * (cur_step + 1)  ### next curriculum -> better model apriory?


        self.ep_infos = {}
        self.ep_infos["ball_rew"] = sum(ball_rew)/len(ball_rew)
        self.ep_infos["pose_rew"] = sum(pose_rew)/len(pose_rew)
        self.ep_infos["l"] = sum(l)/len(l)
        try:
            self.ep_infos["r"] = sum(r) / len(r)
        except ZeroDivisionError:
            self.ep_infos["r"] = 0
        self.ep_infos["curriculum_step"] = cur_step

        # if rew > self.best_mean_reward:
        #         self.best_mean_reward = rew
        #         self.save_model_flag = True

        return rew


class c_TRPO(c_class):
    
    def __init__(self,config, env, video_folder):
        super(c_TRPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        self.algo = TRPO
        self.save_model_flag = True
        self.ep_infos = None
        self.timestamp = 0
        
    def __call__(self):
        env_list = []
        self.model = self.algo(MlpPolicy, self.env ,gamma=self.algo_config["gamma"], vf_stepsize =self.algo_config["learning_rate"],
                              verbose=self.algo_config["verbose"], entcoeff=self.algo_config["ent_coef"], tensorboard_log=self.video_folder,
                              policy_kwargs=dict(net_arch=[dict(pi=[self.algo_config["nn_size"], self.algo_config["nn_size"]],
                                                                vf=[self.algo_config["nn_size"], self.algo_config["nn_size"]])]))
        return self

 
class c_DDPG(c_class):

    def __init__(self,config, env, video_folder):
        super(c_DDPG, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
    
    def __call__(self):
        self.algo = DDPG
        self.model = self.algo(ddpgLnMlpPolicy, self.env, tensorboard_log=self.video_folder)#, verbose=1,random_exploration=0.02,return_range=(-1,1),observation_range=(-1,1))
        return self

class c_TD3(c_class):
    def __init__(self,config, env, video_folder):
        super(c_TD3, self).__init__()
        self.algo_config = load_yaml(config)
        self.algo = TD3
        self.env = env
        self.video_folder = video_folder
        

    
    def __call__(self):
        
        self.model = self.algo(tf3LnMlpPolicy, self.env, verbose=1, tensorboard_log=self.video_folder)#, random_exploration = 0.00, learning_starts=1000, buffer_size=5000)
        return self

    # def learning_callback(self,locals, globals):
    #     if locals["reward"] > self.best_mean_reward:
    #         self.best_mean_reward = locals["reward"]
    #         locals["self"].save(self.video_folder+"best_model.pkl")

