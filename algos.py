from train_utils import  *
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, TD3, TRPO, DDPG, GAIL
from stable_baselines.common import set_global_seeds, make_vec_env

from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.td3.policies import  MlpPolicy as tf3MlpPolicy
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf

class c_class():
    
    def __init__(self):
        self.video_folder = None
        self.model = None
        self.env = None
        self.algo = PPO2
        self.best_mean_reward = 0
        self.ep_infos = None

    def __call__(self):
        raise NotImplementedError

    def validate1(self, model_path=None):
        if model_path == None:
            self.model_path = self.video_folder+"best_model.pkl"
            print ("self.model_path ", self.model_path)
        else:
            self.model_path = model_path
        self.model = self.algo.load(self.model_path)
        obs = self.env.reset()

        for _ in range(1000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            self.env.render()
            try:
                if any(dones) == True:
                    self.env.reset()
            except:
                if dones == True:
                    self.env.reset()
    
    def validate(self):
        obs = self.env.reset()
        obs = self.env.reset()
        rew = []
        pose_rew = []
        ball_rew = []
        l = []
        for i in range(5000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            if dones == True:
                rew.append(info["episode"]["r"])
                obs = self.env.reset()
                pose_rew.append(info["episode"]["pose_rew"])
                ball_rew.append(info["episode"]["ball_rew"])
                l.append(info["episode"]["l"])
                cur_step = info["episode"]["curriculum_step"]
        rew = sum(rew)/len(rew)
        self.ep_infos = {}
        self.ep_infos["ball_rew"] = sum(ball_rew)/len(ball_rew)
        self.ep_infos["pose_rew"] = sum(pose_rew)/len(pose_rew)
        self.ep_infos["l"] = sum(l)/len(l)
        self.ep_infos["curriculum_step"] = cur_step

        if rew > self.best_mean_reward:
                self.best_mean_reward = rew
                self.save_model_flag = True

        return rew

    def learn(self):
        print ("start learning!")
        print (self.algo_config["total_timesteps"])
        self.model.learn(total_timesteps=self.algo_config["total_timesteps"], log_interval=1000, tb_log_name="", callback=self.learning_callback) # 1 6000 000 ~1hr

    def learning_callback(self,locals, globals):
        # print ("inside callback")
        # self.timestamp = locals["self"].num_timesteps
        if  locals["self"].num_timesteps - self.timestamp > self.algo_config["validate_every_timesteps"]:
            self.validate()
            self.timestamp = locals["self"].num_timesteps

        if self.save_model_flag == True:
            self.save_model_flag = False
            locals["self"].save(self.video_folder+"best_model.pkl")
        if self.ep_infos is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/ep_len', simple_value = self.ep_infos["l"] )])
            locals['writer'].add_summary(summary, locals["self"].num_timesteps)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/pose_rew', simple_value=self.ep_infos["pose_rew"] )])
            locals['writer'].add_summary(summary, self.timestamp)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/ball_rew', simple_value=self.ep_infos["ball_rew"] )])
            locals['writer'].add_summary(summary, self.timestamp)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episode_info/curriculum', simple_value=self.ep_infos["curriculum_step"] )])
            locals['writer'].add_summary(summary, self.timestamp)   

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
        env_list= []
        print ("self.algo_config[num_envs] ",self.algo_config["num_envs"])
        if self.algo_config["num_envs"] > 1:
            for _ in range (self.algo_config["num_envs"]):
                env_list.append(self.env)
            self.env = DummyVecEnv(env_list)
        else:
            self.env = self.env
        self.algo = PPO2
        self.model = self.algo(MlpPolicy, self.env, n_steps=self.algo_config["n_steps"], verbose=self.algo_config["verbose"], ent_coef=self.algo_config["ent_coef"],gamma=self.algo_config["gamma"], tensorboard_log=self.video_folder, policy_kwargs=dict(
                    net_arch=[dict(pi=[self.algo_config["nn_size"], self.algo_config["nn_size"]], vf=[self.algo_config["nn_size"], self.algo_config["nn_size"]])]))
        
        
        return self

    def validate(self):
        obs = self.env.reset()
        obs = self.env.reset()
        rew = []
        pose_rew = []
        ball_rew = []
        l = []
        for i in range(5000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            if self.algo_config["num_envs"]> 1:
                is_done = any(dones) == True
            else:
                is_done = dones
            
            if is_done:
                rew.append(info["episode"]["r"])
                obs = self.env.reset()
                pose_rew.append(info["episode"]["pose_rew"])
                ball_rew.append(info["episode"]["ball_rew"])
                l.append(info["episode"]["l"])
                cur_step = info["episode"]["curriculum_step"]
        rew = sum(rew)/len(rew)
        self.ep_infos = {}
        self.ep_infos["ball_rew"] = sum(ball_rew)/len(ball_rew)
        self.ep_infos["pose_rew"] = sum(pose_rew)/len(pose_rew)
        self.ep_infos["l"] = sum(l)/len(l)
        self.ep_infos["curriculum_step"] = cur_step

        if rew > self.best_mean_reward:
                self.best_mean_reward = rew
                self.save_model_flag = True

        return rew

    # def learning_callback(self, locals, globals):
        
    #     #saving best model
    #     mean_rew = sum(locals["returns"])/len(locals["returns"])
    #     if mean_rew > self.best_mean_reward:
    #         self.best_mean_reward = mean_rew
    #         locals["self"].save(self.video_folder+"best_model.pkl")

        
    #     #tf plot episode_average_len, episode_final_dist, curriculum_step
        
    #     av_length = sum([x['l'] for x in locals["ep_infos"]])/len([x['l'] for x in locals["ep_infos"]])
    #     av_final_dist = sum([x['final_dist'] for x in locals["ep_infos"]])/len([x['final_dist'] for x in locals["ep_infos"]])
    #     curriculum = sum([x["curriculum_step"] for x in locals["ep_infos"]])/len([x["curriculum_step"] for x in locals["ep_infos"]])

    #     summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/episode_average_len', simple_value=av_length)])
    #     locals['writer'].add_summary(summary, locals["self"].num_timesteps)
    #     summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/episode_final_dist', simple_value=av_final_dist)])
    #     locals['writer'].add_summary(summary, locals["self"].num_timesteps)
    #     summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/curriculum_step', simple_value=curriculum)])
    #     locals['writer'].add_summary(summary, locals["self"].num_timesteps)            




class c_TRPO(c_class):
    
    def __init__(self,config, env, video_folder):
        super(c_TRPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        self.algo = TRPO
        self.model = self.algo(MlpPolicy, self.env ,gamma=self.algo_config["gamma"], 
                              verbose=self.algo_config["verbose"], entcoeff=self.algo_config["ent_coef"], tensorboard_log=video_folder,
                              policy_kwargs=dict(net_arch=[dict(pi=[self.algo_config["nn_size"], self.algo_config["nn_size"]], 
                                                                vf=[self.algo_config["nn_size"], self.algo_config["nn_size"]])]))

        self.save_model_flag = True
        self.ep_infos = None
        self.timestamp = 0
        
    def __call__(self):
        return self

    



 
class c_DDPG(c_class):

    def __init__(self,config, env, video_folder):
        super(c_DDPG, self).__init__()
        self.video_folder = video_folder
        self.algo = DDPG
        self.env = env
        self.model = self.algo(ddpgMlpPolicy, env, verbose=1, tensorboard_log=video_folder)
    
    def __call__(self):
        return self

    # def learning_callback(self,locals, globals):
    #     if locals["episode_reward"] > self.best_mean_reward:
    #         self.best_mean_reward = locals["episode_reward"]
    #         locals["self"].save(self.video_folder+"best_model.pkl")

class c_TD3(c_class):

    def __init__(self,config, env, video_folder):
        super(c_TD3, self).__init__()
        self.algo = TD3
        self.env = env
        self.model = self.algo(tf3MlpPolicy, env, verbose=1, tensorboard_log=video_folder)
        self.video_folder = video_folder
    
    def __call__(self):
        return self

    # def learning_callback(self,locals, globals):
    #     if locals["reward"] > self.best_mean_reward:
    #         self.best_mean_reward = locals["reward"]
    #         locals["self"].save(self.video_folder+"best_model.pkl")

