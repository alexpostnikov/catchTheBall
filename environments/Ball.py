import numpy as np


class Ball:
    def __init__(self, world, vis, visualizable, config):
        self.ball = world.add_sphere(radius=0.025, mass=0.1)
        # self.ballPose_init = [0.423301, 0.194859, 1.76 - 0.2]
        self.ballPose_init = np.array([0.183301, 0.154859, 1.76])
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
                self.config["environment"]["curruclum"]["ball_pose_distrib"][curriculum_step])
            self.disp = self.disp * (np.random.rand(3) - np.array([0.5, 0.5, 0.5]))

        self.ballPose = self.ballPose_init + self.disp
        self.ball.set_position(self.ballPose)
        self.ball.set_velocity(
            np.array([0.0, 0.0, 0.]), np.array([0.0, 0.0, 0.]))
        return

    @property
    def pose(self):
        return self.ball.get_world_position(0)

    @property
    def velocity(self):
        return self.ball.get_linear_velocity()

    @property
    def pose_scaled(self):
        return self.ball.get_world_position(0) * 0.01

    @property
    def velocity_scaled(self):
        return self.ball.get_linear_velocity() * 0.01

    def set_init_pose(self,pose):
        self.ballPose_init = pose
