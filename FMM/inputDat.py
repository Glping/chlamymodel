"""
module for writing input.dat files for the FMM-BEM
program
"""
import sys
from datetime import datetime
import numpy as np
import FMM.recipe_conversion as recipe_conversion

# import triangulate

def fuseTriangulation(coordinates1, triangulation1, velocities1,
                      coordinates2, triangulation2, velocities2):
    """ fuse coordinates and velocities, and finally triangulation.

    :param coordinates{1,2}: is a list of 3d points, each.
    :param triangulation{1,2}: is list of triples, each. Its values are to be
        interpreted as index of the velocities and coordinates.  Be very careful
        here: ! indices start at 0 !
    :param velocities{1,2}: is a list of vectors, each.
    :return: a dictionary with keys 'coordinates', 'triangulation' and
        'velocities'.
    """
    coordNum1 = len(coordinates1)
    coordNum2 = len(coordinates2)
    triaNum1 = len(triangulation1)
    triaNum2 = len(triangulation2)
    coordinates = np.zeros(coordNum1 + coordNum2)
    velocities = np.zeros(coordNum1 + coordNum2)
    triangulation = np.zeros(triaNum1 + triaNum2)

    for c in range(coordNum1):
        coordinates[c] = coordinates1[c]
        velocities[c] = velocities1[c]
    for c in range(coordNum2):
        coordinates[c + coordNum1] = coordinates2[c]
        velocities[c + coordNum1] = velocities2[c]

    for t in range(triaNum1):
        triangulation[t] = triangulation1[t]
    for t in range(triaNum2):
        triangulation[t + triaNum1] = triangulation2[t] + coordNum1

    return {'coordinates': coordinates,
            'triangulation': triangulation,
            'velocities': velocities }


def inputDat(coordinatesList, triangulationList, velocitiesList, description='descriptive text'):
    """ create the input file for FMM-BEM program. In principle, the program
    can utilize velocities and/or forces, this function only support velocities,
    because it's what we know from experiments. Arguments are lists of lists,
    every sublist represents a different object.

    :param coordinatesList: list of lists of 3d points
    :param triangulation: list of lists of triples. Its values are to
        interpreted as index of the velocities and coordinates.
        Be very careful here: ! indices start at 0 !
    :param velocities: list of lists of 3d vectors
    :return: a triple with the first element being the string to be written
        into the input file for the FMM-BEM program. The second element are the
        ranges of  coordinate indices, belonging to different objects, the third
        element contains range of triangulation indices of different objects.
    """
    # lists of tuples, containing the ranges of objects represented by coordinates and triangulation
    coordinateRanges = []
    triangulationRanges = []

    #f = open('input.dat', 'w')
    coordNum = sum([len(i) for i in coordinatesList])
    triaNum = sum([len(i) for i in triangulationList])
    now = datetime.now()

    # introductory information
    res = description + ',\t' + now.strftime("%A %d. %B %Y, %H:%M:%S") + '\n'
    res += '\t1\t! Problem Type (Do not change this number)\n'
    res += '\t' + str(triaNum) + '\t' + str(coordNum) + '\t1\t'

    # coordinates
    res += '! No. of Elements, Nodes, Mu (Viscosity)\n'
    res += ' $ Nodes (Node #, x, y, and z coordinates):\n'
    c = 1
    cStart = 0
    cEnd = 0
    for coordinates in coordinatesList:
        cStart = c
        for (i,p) in zip(range(len(coordinates)), coordinates):
            res += str(c) + '\t' + str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\n'
            c += 1
        cEnd = c - 1
        coordinateRanges.append((cStart, cEnd))

    # triangulation and velocities
    res += ' $ Elements and Boundary Conditions (Elem #, '
    res += 'Connectivity, BC Type (1=velocity given/2=traction'
    res += ' given, in x,y,z) and given BC Values (in x,y,z)):\n'
    c = 1
    for (rcounter, triangulation, velocities) in zip(range(len(velocitiesList)),triangulationList, velocitiesList):
        cStart = c
        for (j, t) in zip(range(len(triangulation)), triangulation):

            # the triangulation value must be increased by a coordinateRange
            if rcounter != 0:
                t0 = t[0]+1 + coordinateRanges[rcounter-1][1]
                t1 = t[1]+1 + coordinateRanges[rcounter-1][1]
                t2 = t[2]+1 + coordinateRanges[rcounter-1][1]
            else:
                t0 = t[0] + 1
                t1 = t[1] + 1
                t2 = t[2] + 1

            res += str(c) + '\t' + str(t0) + '\t' + str(t1) + '\t' + str(t2)
            # velocities are given for the coordinates, not the triangulated
            # surfaces. let's take the mean.
            # numpy arrays are assumed!
            velocity1 = np.array(velocities[t[0]])
            velocity2 = np.array(velocities[t[1]])
            velocity3 = np.array(velocities[t[2]])
            velo = (velocity1 + velocity2 + velocity3) / 3
            res += '\t1 1 1\t' + str(velo[0]) + '\t' + str(velo[1]) + '\t' + str(velo[2]) + '\n'
            c += 1
        cEnd = c - 1
        triangulationRanges.append((cStart, cEnd))

    return (res, coordinateRanges, triangulationRanges)


def adaptVelocity(coordinates=None,
                  velocities=None,
                  rotation=None,
                  velocity=None,
                  angular=None):
    """
    changes mesh velocities due to the information given in its arguments
    """
    for v in range(len(velocities)):
        for i in range(len(velocities[v])):
            velocities[v][i] = np.dot(rotation, velocities[v][i] + np.cross(np.array(angular), coordinates[v][i])) + np.array(velocity)

def adaptCoordinates(coordinates=None,
                     translation=None,
                     rotation=None):
    """
    change coordinates due to rotation and translation.
    """
    for c in range(len(coordinates)):
        for i in range(len(coordinates[c])):
            coordinates[c][i] = np.dot(rotation, coordinates[c][i]) + translation

def rules2meshes(rules):
    """
    recursively obtain meshes from rules.
    """
    from copy import deepcopy
    res = deepcopy(rules)
    for r in rules:
        if 'type' in rules[r]['system']:
            res[r]['system'] = recipe_conversion.meshProductionRule2Mesh(rules[r]['system'])
        else:
            res[r]['system'] = rules2meshes(rules[r]['system'])
    return res


def meshes2system(meshes):
    """
    recursively generate system.
    """
    from FMM.meshObject import TrivialSystem, ComposedSystem, State
    firstKey = list(meshes.keys())[0]
    mm = meshes[firstKey]
    state = State(translation=mm['translate'],
                  rotation=mm['rotate'],
                  velocity=mm['velocity'],
                  angular=mm['angular'])
    if 'triangulation' in mm['system']:
        return TrivialSystem(coordinates=mm['system']['coordinates'],
                             velocities=mm['system']['velocities'],
                             triangulation=mm['system']['triangulation'],
                             state=state,
                             id=firstKey)
    else:
        return ComposedSystem(state=state,
                              subsystems=[meshes2system({m: mm['system'][m]}) for m in mm['system']],
                              id=firstKey)


def triangulation2file(filename='input.dat',
                       triangulation=None,
                       description='having so much fun!'):
    """
    triangulation is a Triangulation object from FMM.meshObject.
    """
    f = open(filename, 'w')
    now = datetime.now()
    f.write(description + ',\t' + now.strftime("%A %d. %B %Y, %H:%M:%S") + '\n')
    f.write('\t1       ! Problem Type (Do not change this number)\n')
    numTria = len(triangulation.triangulation)
    numCoor = len(triangulation.coordinates)
    f.write('\t{0}\t{1}\t{2}\t ! No. of Elements, Nodes, Mu (Viscosity)\n'.format(numTria, numCoor, 0.85))
    f.write(' $ Nodes (Node #, x, y, and z coordinates):\n')
    for i in range(numCoor):
        f.write('{0}\t{1[0]}\t{1[1]}\t{1[2]}\n'.format(i + 1, triangulation.coordinates[i]))
    f.write(' $ Elements and Boundary Conditions (Elem #, Connectivity, BC Type (1=velocity given/2=traction given, in x,y,z) and given BC Values (in x,y,z)):\n')
    for i in range(numTria):
        jj = triangulation.triangulation[i]
        ii = (int(jj[0]) + 1, int(jj[1]) + 1, int(jj[2]) + 1)
        v1 = triangulation.velocities[int(jj[0])]
        v2 = triangulation.velocities[int(jj[1])]
        v3 = triangulation.velocities[int(jj[2])]
        v = 0.33333333 * (v1 + v2 + v3)
        f.write('{0}\t{1[0]:d}\t{1[1]:d}\t{1[2]:d}\t'.format(i + 1, ii))
        f.write('1\t1\t1\t{0[0]}\t{0[1]}\t{0[2]}\n'.format(v))
    f.close()


def fuse_mesh(system, objectname=None):
    """ given a system, consisting of several meshes, fuse them

    :param system: is a recipe, constructed by
        :func:`FMM.meshObject.transformed_shape`.
    :param objectname: if it is not None, then the system is centered around
        that object.
    :return: an object with:
        self.coordinates
        self.velocities
        self.triangulation
        self.coordRanges
        self.triaRanges
    """
    # for rules2meshes to work, the system must be wrapped up in a dictionary
    # exactly one key value pair
    system = {'all': system}
    # the rules, given as dictionaries with a 'type' key (plane, flagellum,
    # ellipsoid) are transformed to meshes
    meshes = rules2meshes(system)
    # those meshes are then orgaized in a tree structure, given by the
    # FMM.meshObject.System class
    mesh_system = meshes2system(meshes)
    # velocity, angular velocity, rotation and translation of nodes of the
    # tree above are performed.
    mesh_system.adapt()
    # the System tree is fused into a flat array of coordinates, velocities and
    # the triangulation
    return mesh_system.fuse()


def write_input(tria_generator, *tria_generator_args, description='so funny!'):
    """
    tria_generator is a function, that generates a mesh, using the arguments
    tria_generator_args.
    """
    mesh = tria_generator(*tria_generator_args)
    triangulation2file(filename='input.dat', triangulation=mesh,
                       description=description)
    return mesh


def write_remembery(mesh, remembery):
    """
    mesh contains the information about ranges of coordinates and the
    triangulation of a mesh (coordRanges, triaRanges).
    remembery is a file handle.
    """
    remembery.write('coords = {}\n')
    remembery.write('trias = {}\n')
    for k in mesh.coordRanges:
        remembery.write("coords['{0}'] = {1}\n".format(k, mesh.coordRanges[k]))
        remembery.write("trias['{0}'] = {1}\n".format(k, mesh.triaRanges[k]))


def writeInputAndRemembery_NEW(system, remembery=sys.stdout, description='having so much fun'):
    """
    system is a nested dictionary, given by FMM.meshObject.transformedShape.
    key is an identifier and value is a list of systems OR a state and
    mesh production rule.
    """
    mesh = write_input(fuse_mesh, system, description=description)
    # and we need to remember which range of indices refers to which object.
    write_remembery(mesh, remembery)



def writeInputAndRemembery_centered(system, objectname,
        remembery=sys.stdout, description='having so much fun'):
    """
    system is a nested dictionary, given by FMM.meshObject.transformedShape.
    key is an identifier and value is a list of systems OR a state and
    mesh production rule.
    """
    mesh = write_input(lambda syst, objn: fuse_mesh(syst).centered_at(objn),
                       system, objectname)
    # and we need to remember which range of indices refers to which object.
    write_remembery(mesh, remembery)
