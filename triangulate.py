"""
the FMM-BEM program wants triangulated surfaces in its input
files. This module provide them for convex objects and
flagella.
"""

import rotate
import copy
from numpy import array, zeros, pi, dot, cross, arccos
from numpy.linalg import norm
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull


def circleVectors(position, normal, tangent, number, radius):
    """
    a circle with radius, consisting of number vectors at position
    around tangent axis. Start vector is normal.
    """
    res = []
    rotated = copy.deepcopy(normal)
    angle = 2 * pi / number
    for n in range(number):
        rotated = rotate.rotateVector(rotated, angle, tangent)
        rotated = rotated / norm(rotated)
        res.append(position + 0.5 * tangent + rotated * radius)
    return res


def arbitraryNormalFromTangent(tangent):
    """ having a tangetn vector, return some normal vector """
    normal = array([1, 1, 1.0])
    (tx, ty, tz) = tangent
    if tx == 0 and ty == 0:
        normal[2] = 0
    elif tx == 0 and tz == 0:
        normal[1] = 0
    elif ty == 0 and tz == 0:
        normal[0] = 0
    elif tx == 0:
        normal[1] = -tz * normal[2] / ty
    elif ty == 0:
        normal[0] = -tz * normal[2] / tx
    elif tz == 0:
        normal[0] = -ty * normal[1] / tx
    else:
        normal[0] = (ty * normal[1] + tz * normal[2]) / (-tx)
    return normal / norm(normal)


class Flagella(object):
    """
    after intialization of the flagella object, it contains
    the property surfaceParametrization. It is a matrix
    with position vectors as entries. Its indices can be used
    to automatically (via scipy.spatial.Delaunay) construct
    the triangulation.
    """

    def __init__(self, radius, points, nTheta=5):

        points = list(map(lambda v: array(v), points))
        # determine the first normal vector
        tangent = points[1] - points[0]
        normal = arbitraryNormalFromTangent(tangent)
        # self.normals is only used for visualisation purposes
        # it is unrelated to program logic!
        self.normals2draw = [normal]

        self.surfaceParametrization = zeros((len(points)-1, nTheta, 3))
        for (i,p) in zip(range(len(points)-1), points):
            # get all points around p
            for (j,circleVector) in zip(range(nTheta), circleVectors(p, normal, tangent, nTheta, radius)):
                self.surfaceParametrization[i,j] = circleVector

            if i + 2 == len(points):
                break
            # adjust tangent and normal: project old normal onto orthogonal plane.
            tangent = tangent / norm(tangent)
            new_tangent = points[i+2] - points[i+1]
            rot_vec = cross(tangent, new_tangent / norm(new_tangent))
            angle = arccos(dot(tangent, new_tangent / norm(new_tangent)))
            rot_vec = rot_vec / norm(rot_vec) * angle
            normal = dot(rotate.axis2matrix(rot_vec), normal)
            normal = normal/norm(normal)

            tangent = new_tangent
            # normal = normal - tangent*dot(tangent/norm(tangent), normal / norm(normal))
            # self.normals2draw.append(normal)

        self.firstPoint = points[0]
        self.lastPoint = points[-1]

        # self.triangulate()

    def triangulate(self):
        def get2dIndex(point):
            for (c, p) in zip(range(len(twoDpoints)), twoDpoints):
                if p[0] == point[0] and p[1] == point[1]:
                    return c
            print(c,p,point)
            return 'bad result from get2dIndex!'
        (lineElements, circleElements, coordinates) = self.surfaceParametrization.shape
        twoDpoints = [(i,j) for i in range(lineElements) for j in range(circleElements)]
        # the following part of twoDpoints must be at the end of the list
        # so that the 'replace' dictionary, that means the values to be replaced
        # are the highest index values. This ensures that the triangulation contains
        # indices from 1 to index_max.
        twoDpoints += [(i,circleElements) for i in range(lineElements)]

        tri = []
        for (i,j) in [(i,j) for i in range(lineElements-1) for j in range(circleElements)]:
            tri.append((get2dIndex((i, j)), get2dIndex((i+1, j)), get2dIndex((i, j+1))))
            tri.append((get2dIndex((i+1,j)), get2dIndex((i+1,j+1)), get2dIndex((i,j+1))))
        res = []

        # create replace list
        replace = {}
        for (i,p) in zip(range(len(twoDpoints)), twoDpoints):

            if p[1] == circleElements:
                # get the index of p[0] in tri.points
                for (j,ph) in zip(range(len(twoDpoints)), twoDpoints):
                    if ph[0] == p[0] and ph[1] == 0:
                        replace[i] = j

        # replace
        for s in tri:
            (sx, sy, sz) = s
            for r in replace:
                if r == sx:
                    sx = replace[r]
                elif r == sy:
                    sy = replace[r]
                elif r == sz:
                    sz = replace[r]
            res.append((sx,sy,sz))

        # a list of points is needed, so that simplices
        # contains indices of them
        self.surfacePoints = []
        for p in twoDpoints:
            try:
                self.surfacePoints.append(self.surfaceParametrization[p[0],p[1]])
            except:
                continue

        self.simplices = res
        self.points = twoDpoints
        return res

    def triangulateOld(self):
        """ surface triangulation. """
        (lineElements, circleElements, coordinates) = self.surfaceParametrization.shape
        twoDpoints = [(i,j) for i in range(lineElements) for j in range(circleElements)]
        twoDpoints += [(i,circleElements) for i in range(lineElements)]
        tri = Delaunay(twoDpoints)
        #scipy.spatial.delaunay_plot_2d(tri)
        res = []

        # create replace list
        replace = {}
        for (i,p) in zip(range(len(tri.points)), tri.points):

            if p[1] == circleElements:
                # get the index of p[0] in tri.points
                for (j,ph) in zip(range(len(tri.points)), tri.points):
                    if ph[0] == p[0] and ph[1] == 0:
                        replace[i] = j

        # replace
        for s in tri.simplices:
            (sx, sy, sz) = s
            for r in replace:
                if r == sx:
                    sx = replace[r]
                elif r == sy:
                    sy = replace[r]
                elif r == sz:
                    sz = replace[r]
            res.append((sx,sy,sz))

        # a list of points is needed, so that simplices
        # contains indices of them
        self.surfacePoints = []
        for p in tri.points:
            try:
                self.surfacePoints.append(self.surfaceParametrization[p[0],p[1]])
            except:
                continue

        self.points = tri.points
        self.simplices = res

        return res

def sphere(origin, radius, gridLevel=1):
    """
    icosahedron
    """
    from math import sqrt
    phi = 0.5*(1 + sqrt(5))
    ps = []
    length = sqrt(1 + phi*phi)
    for (i,j) in [(i,j) for i in [-1,1] for j in [-phi,phi]]:
        ps.append(array((0,i,j))/length)
        ps.append(array((i,j,0))/length)
        ps.append(array((j,0,i))/length)
    simplices = ConvexHull(ps).simplices

    for c in range(gridLevel-1):
        psNew = []
        for s in simplices:
            direction = ps[s[0]] + ps[s[1]] + ps[s[2]]
            lOld = norm(direction)
            direction = direction/lOld
            psNew.append(direction)
        for p in psNew:
            ps.append(p)
        simplices = ConvexHull(ps).simplices

    ps = [p*radius for p in ps]
    simplices = [(int(a),int(b),int(c)) for (a,b,c) in simplices]

    return (ps, simplices)
