import copy
import itertools
import shutil
import os.path

import extern.FMBEM as FMBEM
import FMM.extractquantities as extr
import RFT.swimmers as swimmers
import numpy as np
import scipy.io
from FMM.meshObject import transformedShape
# pylint: disable=E1101


def realistic_beat_at(time, omega, ds):
    """ receive spermal geometric data """
    head = (5, 2.5)

    data = scipy.io.loadmat(os.path.expanduser('~/.sperm_data/p02F6f05.mat'))
    psi0 = data['psi0'].reshape((44))
    psi1abs = data['psi1abs'].reshape((44))
    phiinterpol = data['phiinterpol'].reshape((44))
    psi = psi0 + psi1abs * 2 * np.sin(omega * time + phiinterpol)

    xflag = -np.cumsum(np.cos(psi)) * ds - head[0]
    yflag = np.cumsum(np.sin(psi)) * ds

    return (xflag, yflag)


def realistic_beat(omega, time_resolution, ds):
    """ receive spermal geometric data """

    T = (2 * np.pi) / omega
    times = np.linspace(0, T, time_resolution)

    head = (5, 2.5)

    data = scipy.io.loadmat(os.path.expanduser('~/.sperm_data/p02F6f05.mat'))
    psi0 = data['psi0'].reshape((44))
    psi1abs = data['psi1abs'].reshape((44))
    phiinterpol = data['phiinterpol'].reshape((44))
    def hfunc(time):
        psi = psi0 + psi1abs * 2 * np.sin(omega * time + phiinterpol)

        x = -np.cumsum(np.cos(psi)) * ds - head[0]
        y = np.cumsum(np.sin(psi)) * ds
        return (x, y)

    flag = [hfunc(t) for t in times]

    return ([h[0] for h in flag], [h[1] for h in flag])


def buildSpermRecipe(timestep,
                     frequency=0.03, spatialResolution=40,
                     headGrid=40, timeResolution=100,
                     length=60, mean_bending_ratio=1):
    """
    return a chlamy recipe. Before passing it to the mesh creation routine,
    another transformedShape wrapper must be executed.
    """
    # TODO: interpolation of the shape, then adapt to length

    ## initialize some constants
    omega = 2 * np.pi * frequency
    dt = 2 * np.pi / (omega * timeResolution)
    ds = length / spatialResolution
    t1 = timestep * dt
    t2 = (timestep + 1) * dt
    head = (5, 2.5)

    ## receive spermal geometric data
    data = scipy.io.loadmat('/home/gary/MPI/data/sperm/p02F6f05.mat')
    psi0 = data['psi0'].reshape((44))
    psi1abs = data['psi1abs'].reshape((44))
    phiinterpol = data['phiinterpol'].reshape((44))
    psi_1 = mean_bending_ratio * psi0 + psi1abs * 2 * np.sin(omega * t1 + phiinterpol)
    psi_2 = mean_bending_ratio * psi0 + psi1abs * 2 * np.sin(omega * t2 + phiinterpol)

    xflag = -np.cumsum(np.cos(psi_1)) * ds - head[0]
    yflag = np.cumsum(np.sin(psi_1)) * ds
    xflagF = -np.cumsum(np.cos(psi_2)) * ds - head[0]
    yflagF = np.cumsum(np.sin(psi_2)) * ds

    ## prepare data structure, that is needed for creating meshes
    ellipsoid = {'type': 'ellipsoid',
                 'position': (0, 0, 0),
                 'lengths': (head[0], head[1], head[1]),
                 'axe1': (1, 0, 0),
                 'axe2': (0, 1, 0),
                 'grid': headGrid}

    positions = [(x, y, 0) for (x, y) in zip(xflag, yflag)]
    positionsF = [(x, y, 0) for (x, y) in zip(xflagF, yflagF)]
    flagellum = {'type': 'flagellum',
                 'positions': positions,
                 'future positions': positionsF,
                 'radius': 0.5,
                 'dt': dt,
                 'azimuth grid': 6}

    res = {'flagellum': transformedShape(flagellum),
           'head': transformedShape(ellipsoid)}

    return res


def buildArtificialSpermRecipe(timestep,
                               frequency=0.03, spatialResolution=40,
                               headGrid=40, timeResolution=100,
                               length=60):
    """
    return a chlamy recipe. Before passing it to the mesh creation routine,
    another transformedShape wrapper must be executed.
    """

    ## initialize some constants
    omega = 2 * np.pi * frequency
    dt = 2 * np.pi / (omega * timeResolution)

    ## receive spermal geometric data
    data1 = swimmers.symmetricSwimmerData(
                dt * timestep, omega, length, spatialResolution)
    head = data1['head']
    xflag = data1['xflag'][:-2] + head[0]
    yflag = data1['yflag'][:-2]
    data2 = swimmers.symmetricSwimmerData(
                dt * (timestep + 1), omega, length, spatialResolution)
    xflagF = data2['xflag'][:-2] + head[0]
    yflagF = data2['yflag'][:-2]

    ## prepare data structure, that is needed for creating meshes
    ellipsoid = {'type': 'ellipsoid',
                  'position': (0, 0, 0),
                  'lengths': (head[0], head[1], head[1]),
                  'axe1': (1, 0, 0),
                  'axe2': (0, 1, 0),
                  'grid': headGrid}

    positions = [(x, y, 0) for (x, y) in zip(xflag, yflag)]
    positionsF = [(x, y, 0) for (x, y) in zip(xflagF, yflagF)]
    flagellum = {'type': 'flagellum',
                 'positions': positions,
                 'future positions': positionsF,
                 'radius': 0.2,
                 'dt': dt,
                 'azimuth grid': 6}

    res = {'flagellum': transformedShape(flagellum),
           'head': transformedShape(ellipsoid)}

    return res


def buildMovingSpermRecipe(timestep,
                           translation=(0, 0, 0),
                           rotation=np.diag([1, 1, 1]),
                           velocity=(0, 0, 0),
                           angular=(0, 0, 0),
                           length=60,
                           frequency=0.03, spatialResolution=40,
                           headGrid=40, timeResolution=100,
                           mean_bending_ratio=1):

    creature = buildSpermRecipe(timestep,
                                frequency=frequency,
                                spatialResolution=spatialResolution,
                                headGrid=headGrid,
                                length=length,
                                timeResolution=timeResolution,
                                mean_bending_ratio=mean_bending_ratio)

    return transformedShape(creature,
                            translation=translation,
                            rotation=rotation,
                            velocity=velocity,
                            angular=angular)
