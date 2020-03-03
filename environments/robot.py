import numpy as np
import raisimpy as raisim

from math import *
import cmath

INIT_POSES =\
[
[1.51429e-09,    -1.56874,  0.00121762,    -1.56881,       -1.57,     1.51676,    0.494994,    0.500008,    0.507557,    0.724445,    0.491012,    0.289995,    0.544989,    0.740545,    0.775573,    0.144698,     0.53565,    0.747655,    0.769407,    0.446968,    0.659029,    0.695925,    0.144999,    0.444984,   0.669998,     0.70015],
[0.106441, -2.02129,  1.39295, -1.83465, -1.78274,  1.44438, 0.494995, 0.500014, 0.507558, 0.724445, 0.492709, 0.289996, 0.544982, 0.740543, 0.775573,  0.14423, 0.540521,  0.74714, 0.770643, 0.444437, 0.661534, 0.697515, 0.144995, 0.444982, 0.669996,  0.70015],
[0.106441, -2.02013,  1.40223,  -1.8332,  -1.7826,  1.44436, 0.494999, 0.385546, 0.391399, 0.558679, 0.476834, 0.289628, 0.447992, 0.608751, 0.637547, 0.177079, 0.610767, 0.734475,  0.70257, 0.554494, 0.604526, 0.696553, 0.144812, 0.486465, 0.641607, 0.670483],
[0.106441, -2.04844,  1.40208, -1.83319, -1.67637,  1.83609, 0.494993, 0.385556, 0.391403, 0.558679,  0.44437, 0.289624, 0.447995, 0.608751, 0.637547, 0.177489, 0.526285, 0.790284, 0.711908, 0.592873, 0.542266, 0.653945, 0.144817, 0.486464, 0.641606, 0.670483],
[0.106441, -2.10218,  1.52627, -2.07416, -1.78249,  1.44434, 0.494992, 0.385559, 0.391375, 0.558614, 0.494978,  0.36135, 0.488083, 0.663226, 0.694599, 0.180673, 0.544981, 0.740543, 0.774407, 0.444979, 0.674996, 0.704295, 0.180673, 0.444982, 0.669997,  0.70015],
[0.106441, -2.04845,  1.40207,  -1.8332,  -1.7826,  1.44436,  0.49499, 0.500024, 0.507569, 0.724464, 0.491921,  0.28963, 0.534108, 0.725768,   0.7601, 0.145077, 0.540551, 0.746798, 0.770777, 0.444046, 0.661991, 0.697836, 0.144812, 0.444982, 0.669996,  0.70015],
[0.0266102,  -2.04845,   1.40207,   -1.8332,   -1.7826,   1.44436,   0.49499,  0.500025,   0.50757,  0.724464,  0.492467,   0.28963,  0.534108,  0.725768,    0.7601,  0.144461,  0.540498,  0.746689,  0.770842,  0.444482,   0.66228,  0.697899,  0.144812,  0.444982,  0.669996,   0.70015],
[0.159661, -2.04845,  1.40207,  -1.8332,  -1.7826,  1.44436,  0.49499, 0.500028, 0.507572, 0.724465, 0.492234, 0.289629,  0.53411, 0.725769,   0.7601, 0.144405, 0.540068, 0.747334, 0.770443, 0.444412, 0.660972, 0.697243, 0.144812, 0.444982, 0.669996,  0.70015],
[0.159661, -2.04834,  1.41957, -1.88786, -1.70297,  1.44436,  0.49499, 0.500021, 0.507567, 0.724463, 0.492279,  0.28963, 0.534108, 0.725768,   0.7601, 0.144437, 0.540291, 0.747253, 0.770527, 0.444301, 0.661186,  0.69737, 0.144812, 0.444981, 0.669996,  0.70015],
[0.159661, -2.04846,  1.41945, -1.88798, -2.10187,  1.44435, 0.494991, 0.500022, 0.507568, 0.724463, 0.492663, 0.289634, 0.534108, 0.725768,   0.7601, 0.144078, 0.540333, 0.747011, 0.770669, 0.444614, 0.661707, 0.697577, 0.144809, 0.444983, 0.669997,  0.70015]]

init_gc_vals = [0, -1.57, 0, -1.57, -1.57, 1.44438,  0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0]

class Robot:
    def __init__(self, world, vis, visualizable, config, resource_directory):
        self.resource_directory = resource_directory
        self.world = world
        self.config = config

        self.ground = self.world.add_ground()
        self.init_robot()
        if visualizable:
            vis = raisim.OgreVis.get()

            self.robot_visual = vis.create_graphical_object(
                self.robot, name="ur10")
            vis.create_graphical_object(
                self.ground, dimension=20, name="floor", material="checkerboard_green")
            vis.select(self.robot_visual[5], False)

        self.minJointsAngles = np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0, 0, 0,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.maxJointsAngles = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.99, 1, 0.99,
                                                                     1.44, 0.98, 0.58, 1.09, 1.36, 1.5, 0.29, 1.09,
                                                                     1.36, 1.49, 0.89, 1.35, 1.42, 0.30, 0.89, 1.34,
                                                                     1.46])

    def transformAction(self, action):
        origin_maximum = self.maxJointsAngles
        origin_minimum = self.minJointsAngles
        maximum = 1.0
        minimum = -1.0
        scale = (origin_maximum - origin_minimum) / (maximum - minimum);
        new_action = origin_minimum + scale * (action - minimum);
        return new_action

    def init_robot(self):
        self.robot = self.world.add_articulated_system(
            self.resource_directory + "ur10_s.urdf")
        self.robot.set_base_position(np.array([0., 0, .30]))
        self.robot.set_control_mode(
            raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)
        self.ground = self.world.add_ground()
        self.world.set_erp(0., 0.)
        self.gc_dim = self.robot.get_generalized_coordinate_dim()
        self.gv_dim = self.robot.get_dof()
        self.num_joints = self.gc_dim  # TODO:
        self.gc, self.gc_init = np.zeros(self.gc_dim), np.zeros(self.gc_dim)
        self.gv, self.gv_init = np.zeros(self.gv_dim), np.zeros(self.gv_dim)

        self.gc_init = np.array(
            INIT_POSES[0])
        self.joint_p_gains = np.array([2000., 2000., 2000., 300., 100., 10., 60, 60, 60,
                                       60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                                       60])
        self.joint_d_gains = np.array([800., 800., 800., 200., 200., 40., 40, 40, 40,
                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])

        self.robot.set_pd_gains(self.joint_p_gains, self.joint_d_gains)
        self.robot.set_generalized_forces(np.zeros(self.gv_dim))
        limits_max = np.array([500, 500, 500, 200, 200, 100, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000])
        limits_min = np.array([-500, -500, -500, -200, -200, -100, -1000, -1000, -1000, -1000, -1000, -1000, -
        1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000])

        # self.minJointsAngles = np.array([-6.28,-6.28,-6.28,-6.28, -6.28, -6.28,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.maxJointsAngles = np.array([6.28, 6.28, 6.28, 6.28, 6.28, 6.28, 0.99, 1, 0.99,
        #                                                              1.44, 0.98, 0.58, 1.09, 1.36, 1.5, 0.29, 1.09,
        #                                                              1.36, 1.49, 0.89, 1.35, 1.42, 0.30, 0.89, 1.34,
        #                                                              1.46])



        self.robot.set_actuation_limits(limits_max, limits_min)
        self.robot.set_generalized_coordinates(self.gc_init)

    def get_endef_pose(self):
        position = self.robot.get_frame_world_position(
            self.robot.get_frame_idx_by_name("svh_f4"))
        rot = self.robot.get_frame_world_rotation_matrix(
            self.robot.get_frame_idx_by_name("svh_f4"))

        # body_index = self.robot.get_body_idx("svh_e2")
        disp = np.dot(rot, np.array([0.0, 0.04, 0.08]).T).T
        return position + disp

    @property
    def endef_pose(self):
        return self.get_endef_pose()

    def get_states(self):
        return self.robot.get_states()

    def reset(self, curriculum_step=0):

        if self.config["environment"]["curruclum"]["use_curruclum"]:
            self.gc_init = INIT_POSES[self.config["environment"]["curruclum"]["pose"][curriculum_step]]
        else:
            self.gc_init = INIT_POSES[1]
        self.robot.set_states(self.gc_init, self.gv_init)
        self.robot.set_generalized_coordinates(self.gc_init)

        self.robot.set_pd_targets(self.gc_init,  self.gv_init)
        return

    def robot_to_init_pose(self):
        self.robot.set_generalized_coordinates(init_gc_vals)


def HTM(i, theta):
    """Calculate the HTM between two links.
    Args:
        i: A target index of joint value.
        theta: A list of joint value solution. (unit: radian)
    Returns:
        An HTM of Link l w.r.t. Link l-1, where l = i + 1.

    """
    # d (unit: mm)
    d1 = 0.1273
    d2 = d3 = 0
    d4 = 0.163941
    d5 = 0.1157
    d6 = 0.0922

    # a (unit: mm)
    a1 = a4 = a5 = a6 = 0
    a2 = -0.612
    a3 = -0.5723

    # List type of D-H parameter
    # Do not remove these
    d = np.array([d1, d2, d3, d4, d5, d6])  # unit: mm
    a = np.array([a1, a2, a3, a4, a5, a6])  # unit: mm
    alpha = np.array([pi / 2, 0, 0, pi / 2, -pi / 2, 0])  # unit: radian

    Rot_z = np.matrix(np.identity(4))
    Rot_z[0, 0] = Rot_z[1, 1] = cos(theta[i])
    Rot_z[0, 1] = -sin(theta[i])
    Rot_z[1, 0] = sin(theta[i])

    Trans_z = np.matrix(np.identity(4))
    Trans_z[2, 3] = d[i]

    Trans_x = np.matrix(np.identity(4))
    Trans_x[0, 3] = a[i]

    Rot_x = np.matrix(np.identity(4))
    Rot_x[1, 1] = Rot_x[2, 2] = cos(alpha[i])
    Rot_x[1, 2] = -sin(alpha[i])
    Rot_x[2, 1] = sin(alpha[i])

    A_i = Rot_z * Trans_z * Trans_x * Rot_x

    return A_i


# Forward Kinematics

def fwd_kin(theta, i_unit='r', o_unit='n'):
    """Solve the HTM based on a list of joint values.
    Args:
        theta: A list of joint values. (unit: radian)
        i_unit: Output format. 'r' for radian; 'd' for degree.
        o_unit: Output format. 'n' for np.array; 'p' for ROS Pose.
    Returns:
        The HTM of end-effector joint w.r.t. base joint
    """

    T_06 = np.matrix(np.identity(4))

    if i_unit == 'd':
        theta = [radians(i) for i in theta]

    for i in range(6):
        T_06 *= HTM(i, theta)

    return T_06


def get_endef_position_by_joint(theta):
    return (fwd_kin(theta)[0:3, 3].T)
