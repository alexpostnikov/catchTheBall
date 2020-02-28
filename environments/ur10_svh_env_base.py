import abc
from abc import abstractclassmethod
import gym
from gym import spaces
import yaml
import numpy as np

import raisimpy as raisim

import collections

# from raisimpy_gym.envs.vis_setup_callback import setup_callback
from datetime import datetime
from .ur10_svh_utils import load_yaml
import os


'''

'''

import numpy as np
import raisimpy as raisim

def normalize(array):
    return np.asarray(array) / np.linalg.norm(array)


def setup_callback():

    vis = raisim.OgreVis.get()

    # light
    light = vis.get_light()
    light.set_diffuse_color(1, 1, 1)
    light.set_cast_shadows(True)
    light.set_direction(normalize([-2., -2., -0.5]))
    vis.set_camera_speed(300)

    # load textures
    vis.add_resource_directory(vis.get_resource_dir() + "/material/checkerboard")
    vis.load_material("checkerboard.material")

    # shadow setting
    manager = vis.get_scene_manager()
    manager.set_shadow_technique(raisim.ogre.ShadowTechnique.SHADOWTYPE_TEXTURE_ADDITIVE)
    manager.set_shadow_texture_settings(2048, 3)

    # scale related settings!! Please adapt it depending on your map size
    # beyond this distance, shadow disappears
    manager.set_shadow_far_distance(3)
    # size of contact points and contact forces
    vis.set_contact_visual_object_size(0.03, 0.2)
    # speed of camera motion in freelook mode
    vis.get_camera_man().set_top_speed(5)






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
    def __init__(self, config, resource_directory,  video_folder=None, visualize=False):
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
        self.visualizable = (self.config["environment"]["render"] or visualize)
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

        self.vis_inited = False
        if self.visualizable:
            self.init_vis()


        self.total_reward = 0.
        self.terminal_reward_coeff, self.done = 0., False
        self.ob_double, self.ob_scaled = np.zeros(
            self.ob_dim), np.zeros(self.ob_dim)
        self.body_linear_vel, self.body_angular_vel = np.zeros(3), np.zeros(3)
        self.reward_buff__ = collections.deque(maxlen=100)
        self.ball_reward_buf = []
        self.pose_reward_buf = []


    def init_ora(self): # ora -> observation reward actions
        raise NotImplementedError
    

    def init_robot(self):
        raise NotImplementedError

    def init_vis(self):
        if not self.vis_inited:
            self.vis_inited = True
            vis = raisim.OgreVis.get()
            self.vis = vis

            vis.set_world(self.world)
            vis.set_window_size(1280, 720)
            vis.set_default_callbacks()

            vis.set_setup_callback(setup_callback)
            vis.set_anti_aliasing(2)
            # vis.init_app()
            self.vis = raisim.OgreVis.get()
            self.vis.set_desired_fps(self.desired_fps)
            self.vis.init_app()
            # self.vis.get_camera_man().set_yaw_pitch_dist(3.14, -1.3, 3, track=True)
