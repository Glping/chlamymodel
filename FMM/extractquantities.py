"""
modify the input system for FMBEM simulations needs a modified
analyzing facility.
Use numpy arrays for speed.
"""
# pylint: disable=E1101

import numpy as np

def force(forces):
    """
    forces: numpy array [number of forces, 3].
    calculate the total force, acting on the body.
    """
    return np.sum(forces, axis=0)


def center(positions):
    """ the center of coordinates """
    return np.mean(positions, axis=0)

def centertorque(positions, forces):
    """
    forces, torques: np.array [number of forces, 3].
    calculate the torque.
    """
    origin = np.mean(positions, axis=0)
    return torque(origin, positions, forces)


def torque(origin, positions, forces):
    """
    forces, torques: np.array [number of forces, 3].
    calculate the torque.
    """
    return np.sum(np.cross(positions - origin, forces), axis=0)


def dissipationdensity(velocities, forces):
    """
    velocities, forces: np.array [number of forces, 3].
    return dissipationdensity, an array of shape (forces.shape[0])
    """
    return np.einsum('ij,ij->i', velocities, forces)


def dissipation(velocities, forces):
    """
    velocities, forces: np.array [number of forces, 3].
    return dissipation, a number.
    """
    return np.sum(velocities * forces)


def monopole(coordinates, forces):
    return np.einsum('ij->j', forces)


def dipole(coordinates, forces):
    return np.einsum('ij,ik->jk', forces, coordinates)


def quadrupole(coordinates, forces):
    return 0.5 * np.einsum('ij,ik,il->jkl', forces, coordinates, coordinates)


def octupole(coordinates, forces):
    return 0.16666 * np.einsum('ij,ik,il,im->jklm',
                     forces,
                     coordinates,
                     coordinates,
                     coordinates)

def hexapole(coordinates, forces):
    return 0.041666 * np.einsum('ij,ik,il,im,in->jklmn',
                     forces,
                     coordinates,
                     coordinates,
                     coordinates,
                     coordinates)

def flowfield(coordinates, forces, eta=1):
    coordinates = np.array(coordinates)
    const = 1 / 8 / np.pi / eta 
    def hfunc(r):
        dr = r - coordinates
        ndr = np.linalg.norm(dr, axis=1)
        fdr = np.array(np.sum(forces * dr, axis=1))
        return  np.sum(np.transpose((np.transpose(np.transpose(dr) * fdr / (ndr * ndr)) + forces)) * const / ndr, axis=1)
    return hfunc
