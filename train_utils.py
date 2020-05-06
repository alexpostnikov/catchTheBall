
import os
import numpy as np
from environments.ur10_svh_utils import load_yaml
best_mean_reward = -np.inf 
rsg_root = os.path.dirname(os.path.abspath(__file__)) + ''
log_dir = rsg_root+"/logs/"
ALGO = "PPO"
cur_dir = rsg_root = os.path.dirname(os.path.abspath(__file__))
config = load_yaml(cur_dir+"/configs/ur10_cfg.yaml")

def check_video_folder(video_folder, unique=True):

    if (video_folder[-1] == "/"):
        video_folder = video_folder[0:-1]

    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    else:
        if unique:
            for i in range(0,9999):
                folder_name = video_folder+"_"+str(i)
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                    video_folder = folder_name
                    return video_folder
    return video_folder


class my_scheduller():
        def __init__(self,init_p,final_p):
            self.init_p = init_p
            self.final_p = final_p
        
        def __call__(self, progress = 0):
            value = self.init_p +  (1 - progress)*(self.final_p-self.init_p)
            return value