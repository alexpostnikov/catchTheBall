import abc
from abc import abstractclassmethod
import gym
from gym import spaces
import yaml
import numpy as np
import raisimpy as raisim
from raisimpy_gym.envs.anymal import ANYMAL_RESOURCE_DIRECTORY
from raisimpy_gym.envs.raisim_gym_env import RaisimGymEnv, keyboard_interrupt
from raisimpy_gym.envs import reward_logger
from raisimpy_gym.envs.vis_setup_callback import setup_callback
from datetime import datetime
from .ur10_svh_utils import load_yaml
import os



schunk_joint_offset = 6

def applayMimic( ptarget, gc_dim=26):
    
    # joint_indexes -= schunk_joint_offset

    for joint in range(schunk_joint_offset,26):
        if joint == 0+schunk_joint_offset: 
            ptarget[joint+4] = ptarget[joint] #10
        if joint == 1+schunk_joint_offset:
            ptarget[joint+1] = ptarget[joint] #8,9
            ptarget[joint+2] = ptarget[joint]

        if joint == 5+schunk_joint_offset: #15,22
            ptarget[joint+4] = ptarget[joint]*0.5
            ptarget[joint+11] = ptarget[joint]*0.5
        
        if joint == 6+schunk_joint_offset: #13,14
            ptarget[joint+1] = ptarget[joint]*1.3588
            ptarget[joint+2] = ptarget[joint]*1.42307
        
        if joint == 10+schunk_joint_offset: #18,17
            ptarget[joint+1] = ptarget[joint]*1.3588
            ptarget[joint+2] = ptarget[joint]*1.42093

        if joint == 14+schunk_joint_offset: #21
            ptarget[joint+1] = ptarget[joint]*1.0434


        if joint == 18+schunk_joint_offset: #25
            ptarget[joint+1] = ptarget[joint]*1.0450

    return ptarget




class ur10SvhBase(gym.Env):
    """Custom Environment that follows gym interface"""
    # metadata = {'render.modes': ['human']}

    def __init__(self, config, resource_directory,  video_folder=None):
        super(ur10SvhBase, self).__init__()
        self.already_closed = False
        self.resource_directory = resource_directory
        self.in_recording = False
        self.video_name = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".mp4"
        self.video_folder = video_folder
        self.curriculum_step = 0
        

        if isinstance(config, str):
            config = load_yaml(config)
        elif not isinstance(config, dict):
            raise TypeError(
                "Expecting the given 'config' argument to be dict or a str (path to the YAML file).")
        self.config = config
        self.visualizable = self.config["environment"]["render"]
        # define other variables

        self.simulation_dt =  config["environment"]["simulation_dt"]#0.0025
        self.control_dt = config["environment"]["control_dt"]
    
        self.max_step = config["environment"]["max_time"] / config["environment"]["control_dt"]

        self.extra_info = dict()  # {str: float}
        self.ob_dim, self.action_dim = 0, 0

        self.visualization_counter = 0
        self.visualize_this_step = self.visualizable
        self.desired_fps = 60.

        self.iteration = 0
        # create world
        self.world = raisim.World()
        self.world.set_time_step(self.simulation_dt)
        self.step_number = 0
        self.terminal_counter = 0
        
        if self.visualizable:
            vis = raisim.OgreVis.get()

            vis.set_world(self.world)
            vis.set_window_size(1280, 720)
            vis.set_default_callbacks()
            vis.set_setup_callback(setup_callback)
            vis.set_anti_aliasing(2)
            try:
                vis.init_app()
            except:
                pass
        # self.check_video_folder()


    
    def init_ora(self): # ora -> observation reward actions
        raise NotImplementedError
    

    def init_robot(self):
        raise NotImplementedError
