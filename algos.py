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

    def __call__(self):
        raise NotImplementedError

    def validate(self, model_path=None):
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


class c_PPO(c_class):
    
    def __init__(self,config, env, video_folder):
        super(c_PPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        env_list= []
        for _ in range (self.algo_config["num_envs"]):
            env_list.append(env)
        self.vec_env = DummyVecEnv(env_list)
        self.algo = PPO2
        self.model = self.algo(MlpPolicy, self.vec_env, n_steps=self.algo_config["n_steps"], verbose=1, ent_coef=self.algo_config["ent_coef"],gamma=self.algo_config["gamma"], tensorboard_log=video_folder, policy_kwargs=dict(
                    net_arch=[dict(pi=[self.algo_config["nn_size"], self.algo_config["nn_size"]], vf=[self.algo_config["nn_size"], self.algo_config["nn_size"]])]))
    def __call__(self):
        return self

    def learning_callback(self, locals, globals):
        
        #saving best model
        mean_rew = sum(locals["returns"])/len(locals["returns"])
        if mean_rew > self.best_mean_reward:
            self.best_mean_reward = mean_rew
            locals["self"].save(self.video_folder+"best_model.pkl")

        
        #tf plot episode_average_len, episode_final_dist, curriculum_step
        
        av_length = sum([x['l'] for x in locals["ep_infos"]])/len([x['l'] for x in locals["ep_infos"]])
        av_final_dist = sum([x['final_dist'] for x in locals["ep_infos"]])/len([x['final_dist'] for x in locals["ep_infos"]])
        curriculum = sum([x["curriculum_step"] for x in locals["ep_infos"]])/len([x["curriculum_step"] for x in locals["ep_infos"]])

        summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/episode_average_len', simple_value=av_length)])
        locals['writer'].add_summary(summary, locals["self"].num_timesteps)
        summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/episode_final_dist', simple_value=av_final_dist)])
        locals['writer'].add_summary(summary, locals["self"].num_timesteps)
        summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward/curriculum_step', simple_value=curriculum)])
        locals['writer'].add_summary(summary, locals["self"].num_timesteps)            




class c_TRPO(c_class):
    
    def __init__(self,config, env, video_folder):
        super(c_TRPO, self).__init__()
        self.algo_config = load_yaml(config)
        self.video_folder = video_folder
        self.env = env
        self.algo = TRPO
        self.model = self.algo(MlpPolicy, self.env, verbose=1, tensorboard_log=video_folder)
        
    def __call__(self):
        return self

    def learning_callback(self,locals, globals):
        mean_rew = sum(locals["reward_buffer"])/len(locals["reward_buffer"])
        if mean_rew > self.best_mean_reward:
            self.best_mean_reward = mean_rew
            locals["self"].save(self.video_folder+"best_model.pkl")


class c_DDPG(c_class):

    def __init__(self,config, env, video_folder):
        super(c_DDPG, self).__init__()
        self.video_folder = video_folder
        self.algo = DDPG
        self.env = env
        self.model = self.algo(ddpgMlpPolicy, env, verbose=1, tensorboard_log=video_folder)
    
    def __call__(self):
        return self

    def learning_callback(self,locals, globals):
        if locals["episode_reward"] > self.best_mean_reward:
            self.best_mean_reward = locals["episode_reward"]
            locals["self"].save(self.video_folder+"best_model.pkl")

class c_TD3(c_class):

    def __init__(self,config, env, video_folder):
        super(c_TD3, self).__init__()
        self.algo = TD3
        self.env = env
        self.model = self.algo(tf3MlpPolicy, env, verbose=1, tensorboard_log=video_folder)
        self.video_folder = video_folder
    
    def __call__(self):
        return self

    def learning_callback(self,locals, globals):
        if locals["reward"] > self.best_mean_reward:
            self.best_mean_reward = locals["reward"]
            locals["self"].save(self.video_folder+"best_model.pkl")

