from numpy.linalg import norm, eig
from numpy import array, zeros, cross, dot, arccos, real, diag
import numpy as np

from math import sin, cos, asin, acos, atan2, sqrt


def angleAxis2Quaternion(alpha,axis):
    """ alpha is a number, axis is a list, tuple or array"""
    axis = array(axis)/norm(array(axis))
    alpha = float(alpha)
    q4 = cos(alpha/2)
    h = sin(alpha/2)
    q1 = h*axis[0]
    q2 = h*axis[1]
    q3 = h*axis[2]
    return (q1,q2,q3,q4)

def quaternion2axis(q1, q2, q3, q4):
    e_w_r = array([q2, q3, q4])
    e_w = e_w_r / norm(e_w_r)
    return 2 * acos(q1) * e_w

def quaternion2EulerAngles(q):
    phi = atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]))
    the = asin(2*(q[0]*q[2]-q[3]*q[1]))
    psi = atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
    return (phi,the,psi)

def angleAxis2EulerAngles(alpha,axis):
    return quaternion2EulerAngles(angleAxis2Quaternion(alpha,axis))

def matrix2axis(matrix):
    return quaternion2axis(*rotation2quaternion(matrix))

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
        print('AAAAAAAAAAAAAAAAAAAAAAAAAA')
    return res

def rotation2quaternion(matrix):
    def way1():
        q4 = 2 * sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
        q1 = 1 / (4 * q4) * (matrix[2, 1] - matrix[1, 2])
        q2 = 1 / (4 * q4) * (matrix[0, 2] - matrix[2, 0])
        q3 = 1 / (4 * q4) * (matrix[1, 0] - matrix[0, 1])
        return (q1, q2, q3, q4)
    def way2():
        q1 = 2 * sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        q2 = 1 / (4 * q1) * (matrix[0, 1] + matrix[1, 0])
        q3 = 1 / (4 * q1) * (matrix[0, 2] + matrix[2, 0])
        q4 = 1 / (4 * q1) * (matrix[2, 1] - matrix[1, 2])
        return (q1, q2, q3, q4)
    def way3():
        q2 = 2 * sqrt(1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])
        q1 = 1 / (4 * q2) * (matrix[0, 1] + matrix[1, 0])
        q3 = 1 / (4 * q2) * (matrix[0, 2] - matrix[2, 0])
        q4 = 1 / (4 * q2) * (matrix[2, 1] + matrix[1, 2])
        return (q1, q2, q3, q4)
    def way4():
        q3 = 2 * sqrt(1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])
        q1 = 1 / (4 * q3) * (matrix[1, 0] - matrix[0, 1])
        q2 = 1 / (4 * q3) * (matrix[0, 2] + matrix[2, 0])
        q4 = 1 / (4 * q3) * (matrix[2, 1] + matrix[1, 2])
        return (q1, q2, q3, q4)
    if 1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2] > 0:
        h1 = 2 * sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    else:
        h1 = 0
    if 1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2] > 0:
        h2 = 2 * sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
    else:
        h2 = 0
    if 1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2] > 0:
        h3 = 2 * sqrt(1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])
    else:
        h3 = 0
    if 1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2] > 0:
        h4 = 2 * sqrt(1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])
    else:
        h4 = 0
    if h1 >= h2 and h1 >= h3 and h1 >= h4:
        return way1()
    if h2 >= h1 and h2 >= h3 and h2 >= h4:
        return way2()
    if h3 >= h2 and h3 >= h1 and h3 >= h4:
        return way3()
    if h4 >= h2 and h4 >= h3 and h4 >= h1:
        return way4()


def axisAndAngleFromMatrix(matrix):
    """
    given a rotation matrix, compute axis of rotation and amount
    of rotation as the length of the rotation axis
    """
    def findArbitraryPerpendicular(vec):
        res = zeros((3))
        for i in [0, 1, 2]:
            if abs(vec[i]) < 0.0000001:
                res[i] = 1
                break
        if abs(res[0]) < 0.00001 and abs(res[1]) < 0.00001 and abs(res[2]) < 0.00001:
            res[0] = -1 / real(vec[0])
            res[1] = -1 / real(vec[1])
            res[2] =  2 / real(vec[2])
        return res

    h = eig(matrix)
    axis = None
    for val, vec in zip(h[0], h[1]):
        if abs(val - 1) < 0.00001:
            axis = vec / norm(vec)

    perp = findArbitraryPerpendicular(axis)
    rotted = dot(matrix, perp)
    angle = arccos(dot(perp / norm(perp), rotted / norm(rotted)))
    return angle * axis

def rotateVector(vector, alpha, axis=(0, 0, 1)):
    """ return a rotated vector by alpha around axis """
    vector = array(vector)
    if abs(norm(array(axis))) < 0.00000000000001:
        return vector
    axis = array(axis)/norm(array(axis))
    rota = rotationMatrix(alpha, axis)
    return dot(rota, vector)
