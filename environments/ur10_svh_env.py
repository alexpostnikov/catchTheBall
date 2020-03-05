#! /usr/bin/python3
import gym
from gym import spaces

import yaml

import numpy as np
from .reward import tolerance
import raisimpy as raisim

from datetime import datetime
from .ur10_svh_env_base import ur10SvhBase
from .ur10_svh_utils import applayMimic
from .Ball import Ball
from .robot import Robot, get_endef_position_by_joint
import math
import time


class ur10svh(ur10SvhBase):
    """Custom Environment that follows gym interface"""

    def __init__(self, config, resource_directory, video_folder=None, visualize=False):

        super(ur10svh, self).__init__(config, resource_directory, video_folder, visualize)
        self.controllable_joints = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        # MUST BE DONE FOR ALL ENVIRONMENTS
        self.update_cur = False
        self.ob_dim = 51  # convention described on top
        self.action_dim = 26 - 11
        self.init_ora()
        self.curriculum_step = self.config["environment"]["curruclum"]["curruclum_step_init"]

        self.desired_fps = 30.
        self.robot = Robot(self.world, raisim.OgreVis.get(), self.visualizable, self.config, resource_directory)
        self.robot.robot_to_init_pose()
        self.ball = Ball(self.world, raisim.OgreVis.get(), self.visualizable, self.config)
        self.goal_pose_init = self.robot.endef_pose
        self.goal_pose = np.array(self.goal_pose_init)

        self.robot.reset()
        self.ball.set_init_pose(self.robot.endef_pose + np.array([0.0, 0.0, 0.02]))
        self.ball.reset()
        self.recording_time_start = time.time()
        self.ee_goal = self.robot.endef_pose

        if self.visualizable:
            self.vis.get_camera_man().set_yaw_pitch_dist(3.14, -1.3, 3, track=True)
            self.vis.add_visual_object("init_pose", "sphereMesh", "white", [0.02, 0.02, 0.02])
            self.vis.add_visual_object("goal_pose", "sphereMesh", "green", [0.02, 0.02, 0.02])
            self.vis.add_visual_object("ee_goal", "sphereMesh", "yellow", [0.02, 0.02, 0.02])

        self.obsEndef = self.robot.get_endef_pose()

        self.p_targets = np.zeros(self.action_dim)
        self.reset()

    def init_ora(self):  # ora -> observation reward actions

        self.action_space = spaces.Box(-1., 1., shape=[self.action_dim])

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=[self.ob_dim])

        self.total_reward = 0.
        self.terminal_reward_coeff, self.done = 0., False
        self.ob_double, self.ob_scaled = np.zeros(
            self.ob_dim), np.zeros(self.ob_dim)

    def step(self, action):
        self.step_number += 1
        self.p_target12 = np.zeros(26)
        skip_counter = 0
        for i in range(self.p_target12.shape[0]):
            if self.controllable_joints[i]:
                self.p_target12[i] = action[i - skip_counter]
            else:
                skip_counter += 1

        p_targets = applayMimic(self.p_target12)
        self.p_targets = self.robot.transformAction(p_targets)
        # self.robot.robot.set_generalized_coordinates(p_targets)
        self.robot.robot.set_pd_targets(self.p_targets, 0 * p_targets)
        self.ee_goal = get_endef_position_by_joint(self.p_targets[0:6])
        self.ee_goal[0, 0] *= -1
        self.ee_goal[0, 1] *= -1
        self.ee_goal[0, 2] += 0.3

        if self.visualizable:
            visual_objects = self.vis.get_visual_object_list()
            visual_objects["ee_goal"].pos_offset = [self.ee_goal[0, 0], self.ee_goal[0, 1], self.ee_goal[0, 2]]
        loop_count = int(self.control_dt / self.simulation_dt + 1.e-10)
        vis_decimation = int(
            1. / (self.desired_fps * self.simulation_dt) + 1.e-10)
        # update world
        for _ in range(loop_count):
            self.world.integrate()
            if self.visualizable and (self.visualization_counter % vis_decimation == 0):
                self.vis.render_one_frame()

                if not self.in_recording:
                    self.recording_time_start = time.time()
                    self.in_recording = True
                    try:
                        self.vis.showWindow()
                    except:
                        pass
                    self.video_name = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + ".mp4"
                    if self.video_folder is not None:
                        self.vis.start_recording_video(
                            self.video_folder + "/" + self.video_name)

            self.visualization_counter += 1

        # update observation
        self.update_observation()

        # update if episode is over or not
        self.is_terminal_state()

        # update reward
        self.update_reward()

        # update extra info
        self.update_extra_info()

        return np.asarray(self.ob_scaled), self.total_reward, self.done, self.extra_info

    def update_observation(self):

        self.gc, self.gv = self.robot.get_states()
        self.ob_double, self.ob_scaled = np.zeros(
            self.ob_dim), np.zeros(self.ob_dim)

        skip = 0
        for i in range(len(self.gc)):
            if self.controllable_joints[i]:
                self.ob_double[3 * (i - skip)] = math.sin(self.gc[i - skip])
                self.ob_double[3 * (i - skip) + 1] = math.cos(self.gc[i - skip])
                self.ob_double[3 * (i - skip) + 2] = self.gv[i - skip] * 0.001
            else:
                skip += 1

        self.ob_double[45:48] = self.ball.pose_scaled
        self.ob_double[48:51] = self.ball.velocity_scaled

        return self.ob_double

    def reset(self):
        self.ball_reward_buf = []
        self.pose_reward_buf = []

        self.robot.reset(self.curriculum_step)

        self.ball.reset(self.curriculum_step)

        self.step_number = 0
        self.update_observation()

        # if self.curriculum_step == 0:
        #
        #     self.ball.ballPose_init[0] = self.goal_pose_init[0]+0.25
        #     self.ball.ballPose_init[1] = self.goal_pose_init[1]
        #     self.ball.ballPose_init[2] = 1.4#self.goal_pose_init[2]

        if self.visualizable:
            l = self.vis.get_visual_object_list()
            l["init_pose"].pos_offset = self.ball.ballPose_init
            l["goal_pose"].pos_offset = self.goal_pose
        return self.ob_scaled

    def render(self):
        if not self.visualizable:
            self.visualizable = True
            self.init_vis()
            try:
                self.vis.showWindow()
            except:
                pass

    def __call__(self):
        return self

    def close(self):

        if not self.already_closed:
            try:
                self.vis.close_app()
            except:
                pass
            self.already_closed = True

    def __del__(self):
        self.close()

    def set_seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def is_terminal_state(self):
        self.done = False
        if (self.step_number >= self.max_step) or (self.ball.pose[2] < 0.4):
            self.terminal_counter += 1
            self.done = True
            if self.visualizable:
                if self.in_recording:
                    if (time.time() - self.recording_time_start) > 7.0:
                        self.visualizable = False
                        if self.video_folder is not None:
                            self.vis.stop_recording_video_and_save()
                            try:
                                self.vis.hideWindow()
                            except:
                                pass
                        self.in_recording = False

            if (self.terminal_counter % self.config["environment"]["render_every_n_resets"] == 0) and (
                    self.visualize_this_step):
                self.terminal_counter = 1
                self.visualizable = True
                try:
                    self.vis.init_app()
                except:
                    pass
        return self.done

    def update_extra_info(self):
        self.extra_info = {}

        self.extra_info["cur_step"] = {
            "r": self.total_reward, "l": self.step_number}
        self.extra_info["r"] = {self.total_reward}
        self.extra_info["l"] = self.step_number
        if self.done:
            self.extra_info["episode"] = {"pose_rew": sum(self.pose_reward_buf) / len(self.pose_reward_buf),
                                          "ball_rew": sum(self.ball_reward_buf) / len(self.ball_reward_buf),
                                          "curriculum_step": self.curriculum_step, "r": self.total_reward,
                                          "l": self.step_number}
            self.extra_info["r"] = {self.total_reward}
            self.extra_info["l"] = self.step_number
            self.extra_info["pose_rew"] = sum(self.pose_reward_buf) / len(self.pose_reward_buf)
            self.extra_info["ball_rew"] = sum(self.ball_reward_buf) / len(self.ball_reward_buf)
            self.extra_info["curriculum_step"] = self.curriculum_step

        return self.extra_info

    def update_reward(self):

        self.obsEndef = self.robot.endef_pose
        catch_dist = np.linalg.norm(self.obsEndef - self.ball.pose)
        ball_reward = tolerance(catch_dist, (0.0, 0.01), 0.05, value_at_margin=0.0001)
        bring_dist = np.linalg.norm(self.ball.pose - self.goal_pose)
        pose_reward = tolerance(bring_dist, (0.0, 0.01), 1.0, value_at_margin=0.00000001)

        self.pose_reward_buf.append(pose_reward)
        self.ball_reward_buf.append(ball_reward)
        self.total_reward = pose_reward * ball_reward

        # print ("self.goal_pose - self.ee_goal ", self.goal_pose - self.ee_goal)
        # t = tolerance(np.linalg.norm(self.goal_pose - self.ee_goal), (0.0, 0.1), 0.2)
        # self.total_reward *= tolerance(np.linalg.norm(self.goal_pose - self.ee_goal), (0.0, 0.1), 0.2)
        # print("self.total_reward ", self.total_reward)

        if self.done:
            self.reward_buff__.append(self.total_reward)
            self.update_curriculum_status()
        return self.total_reward

    def observe(self):
        return self.ob_scaled

    def update_curriculum_status(self):
        if self.update_cur:
            self.update_cur = False
            self.curriculum_step += 1
            self.reward_buff__.clear()
        try:
            if len(self.reward_buff__) > 10 and (sum(self.reward_buff__) / len(self.reward_buff__)) > 0.7:
                if self.curriculum_step < self.config["environment"]["curruclum"]["curruclum_step_max"]:
                    self.curriculum_step += 1
                    self.reward_buff__.clear()
        except ZeroDivisionError:
            pass
        return

    def get_ob_dim(self):
        return self.ob_dim

    def get_action_dim(self):
        return self.action_dim

    def turn_off_visualization(self):
        self.visualizable = False
        if self.video_folder is not None:
            self.vis.stop_recording_video_and_save()
            try:
                self.vis.hideWindow()
            except:
                pass
        self.in_recording = False
