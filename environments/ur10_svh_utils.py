import gym
from gym import spaces
import yaml
import numpy as np
import raisimpy as raisim


from datetime import datetime


def applayMimic( ptarget, gc_dim=26, schunk_joint_offset = 6):

    for joint in range(schunk_joint_offset, gc_dim):
        if joint == 0 + schunk_joint_offset:
            ptarget[joint+4] = ptarget[joint]  # 10

        if joint == 1+schunk_joint_offset:
            ptarget[joint+1] = ptarget[joint]  # 8,9
            ptarget[joint+2] = ptarget[joint]

        if joint == 5+schunk_joint_offset:  # 15,22
            ptarget[joint+4] = ptarget[joint]*0.5
            ptarget[joint+11] = ptarget[joint]*0.5
        
        if joint == 6+schunk_joint_offset:  # 13,14
            ptarget[joint+1] = ptarget[joint]*1.3588
            ptarget[joint+2] = ptarget[joint]*1.42307
        
        if joint == 10+schunk_joint_offset:  # 18,17
            ptarget[joint+1] = ptarget[joint]*1.3588
            ptarget[joint+2] = ptarget[joint]*1.42093

        if joint == 14+schunk_joint_offset:  # 21
            ptarget[joint+1] = ptarget[joint]*1.0434

        if joint == 18+schunk_joint_offset:  # 25
            ptarget[joint+1] = ptarget[joint]*1.0450
    return ptarget



def load_yaml(filename):
    with open(filename, "rb") as f:
        try:
            node = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return node


def get_rotation_matrix_from_quaternion(q):
    """
    Get rotation matrix from the given quaternion.

    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]

    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    w, x, y, z = q
    rot = np.array([[2 * (w**2 + x**2) - 1, 2 * (x*y - w*z), 2 * (x*z + w*y)],
                    [2 * (x*y + w*z), 2 * (w**2 + y**2) - 1, 2*(y*z - w*x)],
                    [2 * (x*z - w*y), 2 * (y*z + w*x), 2 * (w**2 + z**2) - 1]])
    return rot
