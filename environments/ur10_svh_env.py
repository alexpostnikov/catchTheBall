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

rewards = ["ball_dist", "goal_dist", "jerk", "action_dist"] # #

class ur10svh(ur10SvhBase):
    """Custom Environment that follows gym interface"""

    def __init__(self, config, resource_directory, video_folder=None, visualize=False):

        super(ur10svh, self).__init__(config, resource_directory, video_folder, visualize)
        self.controllable_joints = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        self.update_cur = False
        self.ob_dim = 62  # convention described on top
        self.action_dim = 15
        self.init_ora()
        self.curriculum_step = self.config["environment"]["curruclum"]["curruclum_step_init"]

        self.desired_fps = 30.
        self.robot = Robot(self.world, raisim.OgreVis.get(), self.visualizable, self.config, resource_directory)
        self.robot.robot_to_init_pose()
        self.ball = Ball(self.world, raisim.OgreVis.get(), self.visualizable, self.config)
        self.goal_pose_init = self.robot.endef_pose
        self.goal_pose = np.array(self.goal_pose_init)
        self.ball_reward_averaged = 0
        self.pose_reward_averaged = 0

        self.ball.set_init_pose(self.robot.endef_pose + np.array([0.0, 0.0, 0.0]))
        self.ball.reset()
        self.recording_time_start = time.time()
        self.ee_goal = self.robot.endef_pose
        self.update_cur_inner = False

        if self.visualizable:
            self.vis.get_camera_man().set_yaw_pitch_dist(3.14, -1.3, 3, track=True)
            self.vis.add_visual_object("init_pose", "sphereMesh", "white", [0.02, 0.02, 0.02])
            self.vis.add_visual_object("goal_pose", "sphereMesh", "green", [0.02, 0.02, 0.02])
            self.vis.add_visual_object("ee_goal", "sphereMesh", "yellow", [0.02, 0.02, 0.02])

        self.obsEndef = self.robot.get_endef_pose()

        self.p_targets = np.zeros(self.action_dim)
        self.reset()
        self.done = True
        self.p_targets_last = None

    def init_ora(self):  # ora -> observation reward actions

        self.action_space = spaces.Box(-1, 1., shape=[self.action_dim])

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

        self.p_targets = self.robot.transformAction(self.p_target12)
        self.p_targets = applayMimic(self.p_targets)


        self.robot.robot.set_pd_targets(self.p_targets, 0 * self.p_targets)
        # print (self.p_targets[6:])
        # self.robot.robot.set_pd_targets(self.gc, 0 * self.p_targets)
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

                    try:
                        self.vis.showWindow()
                    except:
                        pass
                    self.recording_time_start = time.time()
                    self.in_recording = True
                    self.video_name = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + ".mp4"
                    if self.video_folder is not None:
                        self.vis.start_recording_video(
                            self.video_folder + "/" + self.video_name)

            self.visualization_counter += 1

        # update observation
        self.update_observation()

        # update reward
        self.update_reward()
        self.is_terminal_state()
        self.update_extra_info()
        # update if episode is over or not


        # update extra info

        if any(np.isnan(self.ob_scaled)) == True:
            print ("self.ob_scaled, \n", self.ob_scaled)
            print ("action, \n", action)
            raise


        return np.asarray(self.ob_scaled), self.total_reward, self.done, self.extra_info

    def update_observation(self):

        self.gc, self.gv = self.robot.get_states()
        self.ob_double, self.ob_scaled = np.zeros(
            self.ob_dim), np.zeros(self.ob_dim)

        counter = 0
        for i in range(len(self.gc)):
            if self.controllable_joints[i]:
                self.ob_double[counter] = math.sin(self.gc[i])
                counter += 1
                self.ob_double[counter] = math.cos(self.gc[i])
                counter += 1
                self.ob_double[counter] = self.gv[i] * 0.001
                counter += 1

        self.ob_double[counter:counter+3] = self.ball.pose_scaled
        counter += 3
        self.ob_double[counter:counter+3] = self.ball.velocity_scaled
        counter += 3
        self.ob_double[counter:counter+11] = self.get_ball_collision()
        return self.ob_double

    def reset(self):
        self.p_targets_last = None
        self.robot.reset(self.curriculum_step)

        self.ball.set_init_pose(self.robot.endef_pose)  # + np.array([0.0, 0.0, 0.0 + 0.05*self.curriculum_step]))
        self.ball.reset(self.curriculum_step)

        self.step_number = 0
        self.update_observation()

        if self.visualizable:
            l = self.vis.get_visual_object_list()
            l["init_pose"].pos_offset = self.ball.ballPose
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
        # if (self.step_number >= self.max_step): # or (self.ball.pose[2] < 1.0) or (self.ball.pose[2] > 2.5):
        if (self.step_number >= self.max_step) or (self.ball.pose[2] < 1.0) or (self.ball.pose[2] > 2.5):
            self.terminal_counter += 1
            self.done = True
            self.total_reward += 2 * (self.ball_reward * self.pose_reward)
            self.ball_reward_averaged = np.mean(np.array(self.ball_reward_buf))
            self.pose_reward_averaged = np.mean(np.array(self.pose_reward_buf))


            if self.visualizable:
                if self.in_recording:
                    if (time.time() - self.recording_time_start) > self.config["environment"]["max_time"]:
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
        self.update_curriculum_status()
        return self.done

    def update_extra_info(self):
        self.extra_info = {}
        self.extra_info["cur_step"] = {
            "r": self.total_reward, "l": self.step_number}
        self.extra_info["r"] = {self.total_reward}
        self.extra_info["l"] = self.step_number
        if len(self.pose_reward_buf) == 0:
            self.pose_reward_buf.append(0.)
            self.ball_reward_buf.append(0.)
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
        self.ball_reward = tolerance(catch_dist, (0.0, 0.02), 0.15, value_at_margin=0.0001)

        bring_dist = np.linalg.norm(self.ball.pose - self.goal_pose)
        self.pose_reward = tolerance(bring_dist, (0.0, 0.01), 1.0, value_at_margin=0.001)
        # self.pose_reward = 1.
        self.pose_reward_buf.append(self.pose_reward)
        self.ball_reward_buf.append(self.ball_reward)

        self.total_reward = self.pose_reward  * self.ball_reward

        if "action_dist" in rewards:
            self.ee_rew = tolerance(np.linalg.norm(self.ee_goal - self.goal_pose), (0, 0.05), 2.0,
                                    value_at_margin=0.00000001)
            self.total_reward *= self.ee_rew

        if "jerk" in rewards:
            if self.p_targets_last is not None:
                self.j_rew = tolerance( np.linalg.norm(self.p_targets - self.p_targets_last), (0.0,0.02), 5.0)

                self.total_reward *= self.j_rew
            self.p_targets_last = self.p_targets

        return self.total_reward

    def observe(self):
        return self.ob_scaled

    def update_curriculum_status(self):
        if self.update_cur:
            self.update_cur = False
            if self.config["environment"]["curruclum"]["curruclum_step_max"] != self.curriculum_step:
                self.curriculum_step += 1
                # self.pose_reward_buf.clear()
                self.update_cur_inner = False

        # try:
        #     if ((sum(self.ball_reward_buf) / len(self.ball_reward_buf)) > 0.9) and ((sum(self.pose_reward_buf) / len(self.pose_reward_buf)) > 0.9):
        #         if self.curriculum_step < self.config["environment"]["curruclum"]["curruclum_step_max"]:
        #             self.update_cur_inner = True
        #
        # except ZeroDivisionError:
        # pass

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

    def get_ball_collision(self):

        desired_col_indexes = {26: 0.1, 24: 0.1, 22: 0.1, 20: 0.1, 19: 0.1, 8: 0.1, 6: 0.1, 15: 0.1, 9: 0.1, 11: 0.1, 10: 0.1}

        contacts = self.robot.robot.get_contacts()
        for contact in contacts:

            if contact.get_pair_object_index() != self.robot.index_in_world:

                if contact.get_local_body_index() in list(desired_col_indexes.keys()):
                    # pass
                    desired_col_indexes[contact.get_local_body_index()] = 0.9

        rewards = np.array(list(desired_col_indexes.values()))
        return rewards

