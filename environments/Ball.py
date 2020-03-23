import numpy as np


class Ball:
    def __init__(self, world, vis, visualizable, config):
        self.ball = world.add_sphere(radius=0.025, mass=0.03)
        # self.ball = world.add_sphere(radius=0.015, mass=0.03)
        # self.ballPose_init = [0.423301, 0.194859, 1.76 - 0.2]
        self.ballPose_init = np.array([0.383301, 0.154859, 1.76])
        self.ball.set_position(self.ballPose_init)
        self.ballPose = self.ballPose_init
        self.disp = [0, 0, 0]
        self.config = config
        if visualizable:
            self.robot_visual = vis.create_graphical_object(
                self.ball, name="ball")

    def reset(self, curriculum_step=0):
        if self.config["environment"]["curruclum"]["use_curruclum"]:
            self.disp = np.array(
                self.config["environment"]["curruclum"]["ball_pose_disp_max"][curriculum_step]) - np.array(
                self.config["environment"]["curruclum"]["ball_pose_disp_min"][curriculum_step])
            self.disp = self.disp * (np.random.rand(3)) +  np.array(
                self.config["environment"]["curruclum"]["ball_pose_disp_min"][curriculum_step])
        self.ballPose = self.ballPose_init + self.disp
        self.ball.set_position(self.ballPose)
        self.ball.set_velocity(
            np.array([0.0, 0.0, 0.]), np.array([0.0, 0.0, 0.]))
        return

    @property
    def index_in_world(self):
        for index, object_ in enumerate(self.world.get_object_list()):
            if object_ == self.ball:
                return index

    @property
    def pose(self):
        try:
            return self.ball.get_world_position() 
        except TypeError:
            return self.ball.get_world_position(0)

    @property
    def velocity(self):
        return self.ball.get_linear_velocity()

    @property
    def pose_scaled(self):
        try:
            return self.ball.get_world_position() * 0.01
        except TypeError:
            return self.ball.get_world_position(0) * 0.01

    @property
    def velocity_scaled(self):
        return self.ball.get_linear_velocity() * 0.01

    def set_init_pose(self,pose):
        self.ballPose_init = pose
