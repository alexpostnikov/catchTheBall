import gym
from gym import spaces

import yaml

import numpy as np

import raisimpy as raisim

# from raisimpy_gym.envs.raisim_gym_env import RaisimGymEnv, keyboard_interrupt
# from raisimpy_gym.envs import reward_logger
# from raisimpy_gym.envs.vis_setup_callback import setup_callback
from datetime import datetime
from .ur10_svh_env_base import ur10SvhBase
from .ur10_svh_utils import applayMimic

import collections


class ur10svh(ur10SvhBase):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config, resource_directory, video_folder=None):

        super(ur10svh, self).__init__(config, resource_directory, video_folder)

        self.init_robot()
        # MUST BE DONE FOR ALL ENVIRONMENTS
        self.ob_dim = 6  # convention described on top
        self.action_dim = 7
        self.init_ora()
        self.curriculum_step = self.config["environment"]["curruclum"]["curruclum_step_init"]

        # raisim.gui.reward_logger.init(
        #     ["obsEndef[0]", "obsEndef[1]", "obsEndef[2]"])
        self.desired_fps = 30.
        if self.visualizable:
            vis = raisim.OgreVis.get()
            self.robot_visual = vis.create_graphical_object(
                self.robot, name="ur10")
            vis.create_graphical_object(
                self.ground, dimension=20, name="floor", material="checkerboard_green")
            vis.set_desired_fps(self.desired_fps)
            vis.select(self.robot_visual[0], False)
            vis.get_camera_man().set_yaw_pitch_dist(3.14, -1.3, 3, track=True)

        # define other variables
        self.total_reward = 0.
        self.terminal_reward_coeff, self.done = 0., False
        self.ob_double, self.ob_scaled = np.zeros(
            self.ob_dim), np.zeros(self.ob_dim)
        self.body_linear_vel, self.body_angular_vel = np.zeros(3), np.zeros(3)

        # Define action and observation space
        # They must be gym.spaces objects

        self.obsEndef = self.getEndefPose()

        self.initial_pose = self.getEndefPose()
        self.ball = self.world.add_sphere(radius=0.025, mass=0.1)

        self.ballPose_init = self.getEndefPose() + np.array([0, 0, .5])
        self.ball.set_position(self.ballPose_init)
        self.ballPose = self.ballPose_init
        if self.visualizable:
            self.robot_visual = vis.create_graphical_object(
                self.ball, name="ball")
        self.reward_buff__ = collections.deque(maxlen=20)
        self.ball_reward_buf = []
        self.pose_reward_buf = []

    def init_ora(self):  # ora -> observation reward actions

        self.action_mean = self.gc_init[-self.num_joints:]
        self.action_mean = np.array([*np.ones(6) * 0.0,
                                     0.5])

        self.action_std = np.array([*np.ones(6) * 3.14 * 2,
                                    0.5])

        self.ob_mean = np.array([0.0, 0.0, 2.0,  # gc
                                 0, 0, 2,  # goal
                                 ], dtype=np.float32)
        self.ob_std = np.array([*np.ones(3) * 2.0,  # gc
                                2, 2, 4,  # goal
                                ], dtype=np.float32)

        self.action_space = spaces.Box(-1., 1., shape=[self.action_dim])
        self.observation_space = spaces.Box(low=-2, high=2,
                                            shape=[self.ob_dim])

        self.total_reward = 0.
        self.terminal_reward_coeff, self.done = 0., False
        self.ob_double, self.ob_scaled = np.zeros(
            self.ob_dim), np.zeros(self.ob_dim)

    def init_robot(self):
        self.robot = self.world.add_articulated_system(
            self.resource_directory + "ur10_s.urdf")
        self.robot.set_base_position(np.array([0., 0, 2.0]))
        self.robot.set_control_mode(
            raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)
        self.ground = self.world.add_ground()
        self.world.set_erp(0., 0.)

        self.gc_dim = self.robot.get_generalized_coordinate_dim()
        self.gv_dim = self.robot.get_dof()
        self.num_joints = self.gc_dim  # TODO:
        self.gc, self.gc_init = np.zeros(self.gc_dim), np.zeros(self.gc_dim)
        self.gv, self.gv_init = np.zeros(self.gv_dim), np.zeros(self.gv_dim)

        self.p_targets, self.v_targets = np.zeros(
            self.gc_dim), np.zeros(self.gv_dim)
        self.p_target12 = np.zeros(self.num_joints)
        self.gc_init = np.ones(self.gc_dim) * 0.1
        # self.gc_init[0:6] = np.array([0, -1, 1.5, 1, 0., 3.14])

        self.gc_init[0:6] = np.array([0.000136, -0.000390, -0.000224, 0.000961, 0.000392, 0.000282]) * 3.625574
        self.gc_init = np.array([0.159661, - 2.04846,1.41945, - 1.88798, - 2.10187,  1.44435,  0.494991,  0.500022,   0.507568,  0.724463,   0.492663,   0.289634,   0.534108,   0.725768, 0.7601,   0.144078, 0.540333, 0.747011, 0.770669, 0.444614, 0.661707, 0.697577, 0.144809, 0.444983, 0.669997, 0.70015])
        self.joint_p_gains = np.array([2000., 2000., 2000., 300., 100., 10., 600, 600, 600,
                                       600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
                                       600])
        self.joint_d_gains = np.array([800., 800., 800., 200., 200., 40., 40, 40, 40,
                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])

        self.robot.set_pd_gains(self.joint_p_gains, self.joint_d_gains)
        self.robot.set_generalized_forces(np.zeros(self.gv_dim))
        limits_max = np.array([500, 500, 500, 200, 200, 100, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000])
        limits_min = np.array([-500, -500, -500, -200, -200, -100, -1000, -1000, -1000, -1000, -1000, -1000, -
        1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000])

        self.robot.set_actuation_limits(limits_max, limits_min)
        self.robot.set_generalized_coordinates(self.gc_init)

    def getBallPose(self):
        return self.ball.get_world_position(0)

    def getEndefPose(self):
        position = self.robot.get_frame_world_position(
            self.robot.get_frame_idx_by_name("svh_f4"))
        rot = self.robot.get_frame_world_rotation_matrix(
            self.robot.get_frame_idx_by_name("svh_f4"))
        disp = np.dot(rot, np.array([0.0, 0.0, 0.08]).T).T

        return position + disp

    def step(self, action):
        self.step_number += 1
        # action scaling
        self.p_target12 = np.array(
            action, dtype=np.float64) + 0.05 * np.random.rand(action.shape[0])

        self.p_target12 *= self.action_std
        self.p_target12 += self.action_mean

        if self.p_target12[6] > 0.5:
            self.p_target12_ext = np.array([*self.p_target12[0:6], *(np.ones(20)*0.9)])
        if self.p_target12[6] < 0.5:
            self.p_target12_ext = np.array([*self.p_target12[0:6], *(np.ones(20) * 0.1)])

        self.p_targets = applayMimic(self.p_target12_ext)
        self.robot.set_pd_targets(self.p_targets, self.v_targets)

        loop_count = int(self.control_dt / self.simulation_dt + 1.e-10)
        vis_decimation = int(
            1. / (self.desired_fps * self.simulation_dt) + 1.e-10)

        # update world
        for _ in range(loop_count):
            self.world.integrate()
            if self.visualizable and (self.visualization_counter % vis_decimation == 0):
                vis = raisim.OgreVis.get()
                vis = raisim.OgreVis.get()

                vis.render_one_frame()

                if (not self.in_recording):
                    self.in_recording = True
                    self.video_name = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + ".mp4"
                    if self.video_folder is not None:
                        vis.start_recording_video(
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

        self.ballPose = self.getBallPose()

        self.prevObsEndef = self.obsEndef
        self.obsEndef = self.getEndefPose()
        self.ob_double[0:3] = self.obsEndef
        self.ob_double[3:6] = self.ballPose
        # print ("observations: ", self.ob_double)
        self.ob_scaled = np.array(
            (self.ob_double - self.ob_mean) / self.ob_std, dtype=np.float32)

        return self.ob_scaled

    def reset(self):
        self.ball_reward_buf = []
        self.pose_reward_buf = []

        self.robot.set_states(self.gc_init, self.gv_init)
        self.robot.set_generalized_coordinates(self.gc_init)
        self.reset_ball()
        self.step_number = 0
        self.update_observation()

        return self.ob_scaled

    def render(self, mode='human'):
        self.visualizable = True

    def __call__(self):
        return self

    def close(self):
        if not self.already_closed:
            try:
                vis = raisim.OgreVis.get()
                vis.close_app()
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
        if (self.step_number >= self.max_step) or (self.getBallPose()[2] < 0.5):
            self.terminal_counter += 1
            self.done = True
            if self.visualizable:
                self.visualizable = False
                try:
                    vis.close_app()
                except:
                    pass
                vis = raisim.OgreVis.get()
                if self.video_folder is not None:
                    vis.stop_recording_video_and_save()
                self.in_recording = False

            if (self.terminal_counter % self.config["environment"]["render_every_n_resets"] == 0) and (
            self.visualize_this_step):
                self.terminal_counter = 1
                self.visualizable = True
                try:
                    vis.init_app()
                except:
                    pass
        return self.done

    def update_extra_info(self):
        self.extra_info = {}
        self.extra_info["cur_step"] = {
            "r": self.total_reward, "l": self.step_number}
        if (self.done):
            self.extra_info["episode"] = {"pose_rew": sum(self.pose_reward_buf) / len(self.pose_reward_buf),
                                          "ball_rew": sum(self.ball_reward_buf) / len(self.ball_reward_buf),
                                          "curriculum_step": self.curriculum_step, "r": self.total_reward,
                                          "l": self.step_number}

        return self.extra_info

    def update_reward(self):

        self.ball_reward = np.exp(
            (- 4 * np.linalg.norm(self.obsEndef - self.ballPose)))

        if np.linalg.norm(self.obsEndef - self.ballPose) < 0.05:
            self.ball_reward = 1.
        self.ball_reward_buf.append(self.ball_reward)

        self.bring_dist = self.obsEndef - np.array([0.2, 0.2, 2.7])
        self.pose_reward = np.exp(
            (- 4 * np.linalg.norm(self.bring_dist)))
        if np.linalg.norm(self.bring_dist) < 0.05:
            self.pose_reward = 1.
        self.pose_reward_buf.append(self.pose_reward)

        self.total_reward = self.pose_reward * self.ball_reward

        if self.done:
            self.reward_buff__.append(self.pose_reward)
            self.update_curriculum_status()

        return self.total_reward

    def observe(self):
        return self.ob_scaled

    def reset_ball(self):
        self.disp = [0, 0, 0]
        if self.config["environment"]["curruclum"]["use_curruclum"]:
            self.disp = np.array(
                self.config["environment"]["curruclum"]["ball_pose_distrib"][self.curriculum_step])
            self.disp = self.disp * (np.random.rand(3) - np.array([0.5, 0.5, 0.5]))

        self.ballPose = self.ballPose_init + self.disp
        # print("new ball pose: ", self.ballPose)
        self.ball.set_position(self.ballPose)
        self.ball.set_velocity(
            np.array([0.0, 0.0, 0.]), np.array([0.0, 0.0, 0.]))
        return

    def update_curriculum_status(self):
        try:
            if len(self.reward_buff__) > 10 and (sum(self.reward_buff__) / len(self.reward_buff__)) > 0.9:
                if self.curriculum_step < self.config["environment"]["curruclum"]["curruclum_step_max"]:
                    self.curriculum_step += 1
                    self.reward_buff__.clear()
        except ZeroDivisionError:
            pass
        return
