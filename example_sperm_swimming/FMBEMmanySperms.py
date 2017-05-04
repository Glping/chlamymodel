#!/usr/bin/env python3
"""
call this program in a folder with a config.py file in it. It must contain
the following variables:
    processes = 8
    timestep = [0, 0]
    origin = [(0, 0, -10), (0, 0, 10)]
    rotation = [np.diag([1, 1, 1]), np.diag([1, 1, 1])]
    spatialResolution = 30
    bodyGrid = 35
    timeResolution = 100
    frequency = 0.03
Some reasonable values are shown, which are not default values. They must be
set explicitly!
"""
import sys
import os

import FMM.runManySwimmers as runManySwimmers
import FMM.runSperm as runSperm
import FMM.handlerecipes as handlerecipes

import numpy as np


def axis2matrix(axis):
    return rotationMatrix(norm(axis), axis)

def rotationMatrix(alpha, axis):
    """ return the rotation matrix, given axis and angle """
    if alpha < 0.00001:
        return diag([1, 1, 1])
    (a, b, c, d) = angleAxis2Quaternion(alpha, axis)
    res = zeros((3, 3))
    res[0,0] = -1 + 2*a*a + 2*d*d
    res[1,1] = -1 + 2*b*b + 2*d*d
    res[2,2] = -1 + 2*c*c + 2*d*d
    res[0,1] = 2*(a*b - c*d)
    res[0,2] = 2*(a*c + b*d)
    res[1,2] = 2*(b*c - a*d)
    res[1,0] = 2*(a*b + c*d)
    res[2,0] = 2*(a*c - b*d)
    res[2,1] = 2*(b*c + a*d)
    if abs(np.linalg.det(res) - 1) > 0.01:
        print(np.linalg.det(res))
        print('cannot create rotation matrix')
    return res



sys.path.append(os.getcwd())
from config import processes, timestep, origin, rotation, length
from config import spatialResolution, headGrid, timeResolution, frequency
from config import timestepsH
try:
    from config import mean_bending_ratio
except:
    mean_bending_ratio = 1

partnum = len(timestep)
velocity = [(0, 0, 0)] * partnum
angular = [(0, 0, 0)] * partnum

## write those to files and use them for lab frame creation
trajectories = np.zeros((partnum, timestepsH + 1, 3))
rotations = np.zeros((partnum, timestepsH + 1, 9))
velocities = np.zeros((partnum, timestepsH, 3))
angulars = np.zeros((partnum, timestepsH, 3))

for pi in range(partnum):
    trajectories[pi, 0, :] = np.array(origin[pi])
    rotations[pi, 0, :] = rotation[pi].reshape((9))

## some constants, then run timesteps
T = 1 / frequency
dt = T / timeResolution

spermnames = ['chlamy{0}'.format(i) for i in range(partnum)]
for i in range(timestepsH):

    sperms = {spermnames[pi]: runSperm.buildMovingSpermRecipe(
                                     timestep[pi] + i,
                                     translation=origin[pi],
                                     rotation=rotation[pi],
                                     frequency=frequency,
                                     length=length,
                                     spatialResolution=spatialResolution,
                                     headGrid=headGrid,
                                     timeResolution=timeResolution,
                                     mean_bending_ratio=mean_bending_ratio)
                     for pi in range(partnum)}

    (F, friction) = runManySwimmers.run_timestep(
                        timestep[0] + i,
                        handlerecipes.transformed_shape(sperms),
                        spermnames,
                        processes=processes,
                        removefolder=False)

    V = np.linalg.lstsq(friction, F)[0]

    for pi in range(partnum):
        velocity[pi] = -V[3 * pi : 3 * (pi + 1)]
        angular[pi] = -V[3 * (partnum + pi): 3 * (partnum + pi + 1)]
        origin[pi] = origin[pi] + velocity[pi] * dt
        rotation[pi] = np.dot(axis2matrix(dt * angular[pi]), rotation[pi])
        trajectories[pi, i + 1, :] = np.array(origin[pi])
        velocities[pi, i, :] = velocity[pi]
        rotations[pi, i + 1, :] = rotation[pi].reshape((9))
        angulars[pi, i, :] = angular[pi]


for pi in range(partnum):
    np.savetxt('positions_{0}'.format(pi + 1), trajectories[pi])
    np.savetxt('rotations_{0}'.format(pi + 1), rotations[pi])
    np.savetxt('velocities_{0}'.format(pi + 1), velocities[pi])
    np.savetxt('angulars_{0}'.format(pi + 1), angulars[pi])
