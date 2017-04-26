"""
read the output.dat file, produced by the BEM-FMM program.
do several analizes
- flowfield
- forces
- torques
"""
import os
import sys
import copy

from numpy import array, dot, cross, sqrt
from numpy.linalg import norm
import numpy as np

from FMM.analyse import force, torque, flowfield

def readMeshFromInput(filename='input.dat'):
    """
    read points and triangulation and return it.
    """
    f = open(filename, 'r')
    points = []
    trias = []
    pointFlag = False
    triaFlag = False
    for line in f:
        if pointFlag:
            try:
                hh = line.split()[1:]
                points.append([float(h) for h in hh])
            except:
                pointFlag = False
                triaFlag = True
        elif triaFlag:
            hh = line.split()[1:4]
            trias.append([int(h) for h in hh])
        elif '$ Nodes' in line:
            pointFlag = True
    f.close()
    return (points, trias)


def collectData(folder):
    """
    given a folder, read data with the help of a local remembery module.
    Forces, velocities and coordinates and areas of triangle surfaces are
    returned in a POD.
    """
    oldFolder = os.getcwd()
    os.chdir(folder)
    oldpath = sys.path
    sys.path.append(os.getcwd())
    from remembery import coordinateRange, triangulationRange
    forces = {}
    velocities = {}
    coordinates = {}
    areas = {}

    for (c, t) in zip(coordinateRange, triangulationRange):
        h = velocitiesForcesPositions('input.dat', 'output.dat', c, t)
        for i in h['positions']:
            forces[i] = h['forces'][i]
            velocities[i] = h['velocities'][i]
            coordinates[i] = h['positions'][i]
            areas[i] = h['areas'][i]

    os.chdir(oldFolder)
    sys.path = oldpath
    class Res(object):
        def __init__(self):
            self.forces = forces
            self.velocities = velocities
            self.coordinates = coordinates
            self.areas = areas
    return Res()


def triangulation(filename, posiRange, triaRange):
    """
    filename is the name of the input file.
    """
    f = open(filename, 'r')
    tria = {}
    posi = {}
    triaFlag = False
    posiFlag = False
    for line in f:
        if '$ Nodes (Nod' in line:
            posiFlag = True
        if '$ Elements and Boundary Co' in line:
            triaFlag = True
            posiFlag = False
        if posiFlag:
            try:
                (a,b,c,d) = line.split()
            except:
                continue
            if int(a) >= posiRange[0] and int(a) <= posiRange[1]:
                posi[int(a)] = (float(b), float(c), float(d))

        if triaFlag:
            try:
                (a,b,c,d,e,f,g,h,i,j) = line.split()
            except:
                continue
            if int(a) >= triaRange[0] and int(a) <= triaRange[1]:
                tria[int(a)] = (int(b), int(c), int(d))

    return (posi, tria)



def triangulationFromInputfile(filename, posiRange, triaRange):
    """
    filename is the name of the input file. (usually 'input.dat')
    posiRange: indices of coordinates.
    triaRange: indices of triangulation.
    return two numpy arrays, containing the coordinates and the triangulation
    with respect to those coordinates.
    """
    f = open(filename, 'r')
    tria = np.zeros((triaRange[1] - triaRange[0] + 1, 3), dtype=int)
    posi = np.zeros((posiRange[1] - posiRange[0] + 1, 3))
    triaFlag = False
    posiFlag = False
    for line in f:
        if '$ Nodes (Nod' in line:
            posiFlag = True
        if '$ Elements and Boundary Co' in line:
            triaFlag = True
            posiFlag = False
        if posiFlag:
            try:
                (a,b,c,d) = line.split()
            except:
                continue
            if int(a) > posiRange[0] and int(a) <= posiRange[1] + 1:
                posi[int(a) - 1 - posiRange[0]] = (float(b), float(c), float(d))

        if triaFlag:
            try:
                (a,b,c,d,e,f,g,h,i,j) = line.split()
            except:
                continue
            if int(a) > triaRange[0] and int(a) <= triaRange[1] + 1:
                tria[int(a) - 1 - triaRange[0]] = (int(b) - 1 - posiRange[0],
                                                   int(c) - 1 - posiRange[0],
                                                   int(d) - 1 - posiRange[0])

    return (posi, tria)


def triangulationFromInputNameByName(objectspecifier,
        folder='.',
        filename='input.dat'):
    oldpath = copy.deepcopy(sys.path)
    sys.path.append(folder)
    import remembery
    res = triangulationFromInputfile('{0}/{1}'.format(folder, filename),
                      remembery.coords[objectspecifier],
                      remembery.trias[objectspecifier])
    sys.path = oldpath
    return res


def triangulationFromInputAll(folder='.', filename='input.dat'):
    return triangulationFromInputNameByName('all',
                             folder=folder,
                             filename=filename)


def triangleArea(v1, v2, v3):
    """
    given three position vectors, calculate the area of a triangle, using
    Heron's formula.
    """
    [v1, v2, v3] = [ array(v) for v in [v1, v2, v3] ]
    [a, b, c] = [ norm(d) for d in [v1-v2, v2-v3, v3-v1] ]
    s = (a+b+c)/2.0
    A = sqrt(s*(s-a)*(s-b)*(s-c))
    return A


def triangleAreas(filename, posiRange, triaRange):
    """
    filename is the input filename.
    """
    (posi, tria) = triangulation(filename, posiRange, triaRange)
    areas = {}
    for t in tria.keys():
        areas[t] = triangleArea(posi[tria[t][0]], posi[tria[t][1]], posi[tria[t][2]])
    return areas


def triangleAreasFromInputfile(filename, posiRange, triaRange):
    """
    filename is the input filename.
    """
    (posi, tria) = triangulationFromInputfile(filename, posiRange, triaRange)
    areas = np.zeros((triaRange[1] - triaRange[0] + 1))
    for (i, t) in enumerate(tria):
        areas[i] = triangleArea(posi[t[0]], posi[t[1]], posi[t[2]])
    return areas


def extractData(posiRange, triaRange,
                infile='input.dat',
                outfile='output.dat'):
    """
    given a number range, that correspond to an object,
    read velocities, forces and positions and areas as np.arrays.
    """
    areas = triangleAreasFromInputfile(infile, posiRange, triaRange)

    f = open(outfile, 'r')
    num = triaRange[1] - triaRange[0] + 1
    velocities = np.zeros((num, 3))
    forces = np.zeros((num, 3))
    positions = np.zeros((num, 3))
    for line in f:
        try:
            (a,b,c,d,e,f,g,h,i,j) = line.split()
            index = int(a)
            if index > triaRange[0] and index <= triaRange[1] + 1:
                velocities[index - triaRange[0] - 1] = (float(b), float(c) ,float(d))
                forces[index - triaRange[0] - 1] = (float(e), float(f) ,float(g))
                positions[index - triaRange[0] - 1] = (float(h), float(i) ,float(j))
            if index > triaRange[1]:
                break
        except:
            continue

    for k in range(num):
        forces[k] *= areas[k]

    class Res(object):
        def __init__(self):
            self.forces = forces
            self.velocities = velocities
            self.coordinates = positions
            self.areas = areas
    return Res()


def extractDataByName(objectspecifier,
                      folder='.',
                      infile='input.dat',
                      outfile='output.dat'):
    oldpath = copy.deepcopy(sys.path)
    sys.path.append(folder)
    import remembery
    import imp
    imp.reload(remembery)
    res = extractData(remembery.coords[objectspecifier],
                      remembery.trias[objectspecifier],
                      infile='{0}/{1}'.format(folder, infile),
                      outfile='{0}/{1}'.format(folder, outfile))
    sys.path = oldpath
    return res


def extractAllData(folder='.', infile='input.dat', outfile='output.dat'):
    return extractDataByName('all',
                             folder=folder,
                             infile=infile,
                             outfile=outfile)


def velocitiesForcesPositions(infile, outfile, posiRange, triaRange):
    """
    filename is the output file.
    given a number range, that correspond to an object,
    read velocities, forces and positions.
    """

    areas = triangleAreas(infile, posiRange, triaRange)

    f = open(outfile, 'r')
    velocities = {}
    forces = {}
    positions = {}
    for line in f:
        try:
            (a,b,c,d,e,f,g,h,i,j) = line.split()
            index = int(a)
            if index >= triaRange[0] and index <= triaRange[1]:
                velocities[index] = (float(b), float(c) ,float(d))
                forces[index] = (float(e), float(f) ,float(g))
                positions[index] = (float(h), float(i) ,float(j))
            if index == triaRange[1]:
                break
        except:
            continue

    for k in forces.keys():
        f1 = forces[k][0] * areas[k]
        f2 = forces[k][1] * areas[k]
        f3 = forces[k][2] * areas[k]
        forces[k] = (f1, f2, f3)
    return {'velocities': velocities,
            'forces': forces,
            'positions': positions,
            'areas': areas }


# the following function are for testing purpose.
# after running a simulation, it is not possible to see
# what we were simulating. Let's uncover:

def plotObject(filename, positions):
    """
    point plots of the positions.
    filename should not have a file ending.
    """
    from plotting.plot2d import points
    xx = [x for (x,y,z) in positions.values()]
    yy = [y for (x,y,z) in positions.values()]
    zz = [z for (x,y,z) in positions.values()]
    data = {'x':xx,'y':yy,'z':zz}
    for (i,j) in [ (i,j) for i in data.keys() for j in data.keys() if i < j ]:
        mn1 = min(data[i])
        mn2 = min(data[j])
        mx1 = max(data[i])
        mx2 = max(data[j])
        points(data[i], data[j], xrange=(mn1,mx1), yrange=(mn2,mx2),
               xlabel=i+'-coordinate', ylabel=j+'-coordinate', filename=filename+'-'+i+j+'.pdf')

def plotForces(filename, positions, forces):
    """
    point plots of forces and positions.
    filename should not have a file ending.
    """
    from plotting.plot2d import points
    xx = [x for (x,y,z) in positions.values()]
    yy = [y for (x,y,z) in positions.values()]
    zz = [z for (x,y,z) in positions.values()]
    fx = [x for (x,y,z) in forces.values()]
    fy = [y for (x,y,z) in forces.values()]
    fz = [z for (x,y,z) in forces.values()]
    spatialData = {'x':xx,'y':yy,'z':zz}
    forceData = {'x':fx,'y':fy,'z':fz}
    for p in spatialData:
        for f in forceData:

            mnp = min(spatialData[p])
            mnf = min(forceData[f])
            mxp = max(spatialData[p])
            mxf = max(forceData[f])
            points(spatialData[p], forceData[f], xrange=(mnp,mxp), yrange=(mnf,mxf),
                   xlabel=p+'-coordinate', ylabel='force in '+f+' direction', filename=filename+'-'+p+'F'+f+'.pdf')

def plotFlowfield(filename, positions, forces):
    """
    plot the flowfield via stokeslet integration
    """
    import plotting.stream as plt
    import plotting.preparation as prep
    field = flowfield(positions, forces)
    xx = [x for (x,y,z) in positions.values()]
    yy = [y for (x,y,z) in positions.values()]
    zz = [z for (x,y,z) in positions.values()]
    minimum = min([min(i) for i in [xx,yy,zz]])
    maximum = max([max(i) for i in [xx,yy,zz]])
    (x,y,u,v) = prep.prepare3dPlot(field, rangex=(minimum-5,maximum+5), rangey=(minimum-5,maximum+5), grid=(20,20))
    plt.streamsContour(x,y,u,v, filename=filename + '.pdf', minimum=-1)

def lists2file(filename, ll):
    f = open(filename, 'w')
    for i in range(len(ll[0])):
        for l in ll:
            f.write(str(l[i]) + ' ')
        f.write('\n')
    f.close()

def unzip(lll):
    """
    list of tuples -> tuples (not really, they are lists) of lists
    """
    num = len(lll[0])
    res = []
    for i in range(num):
        res.append([])
    for l in lll:
        for i in range(num):
            res[i].append(l[i])
    return res

def concatData(data):
    """
    given a list of dictionaries, return a union of them.
    Undefined behaviour for multiple occurences of a key.
    """
    res = {}
    for d in data:
        for key in d:
            res[key] = d[key]
    return res

def analyze(infile, outfile, file2write, objects, posiRanges, triaRanges, outConf):
    """
    infile: the input file for FMBEM
    outfile: the output file of FMBEM
    file2write: part of the output filenames used by this function
    posiRanges: ranges, representing the coordinates of different objects
    triaRanges: ranges, representing the triangulation of different objects
    outConf: a dictionary that specifies which analyzer calculations are to be done.
    """
    data = [ velocitiesForcesPositions(infile, outfile, pR, tR) for (pR, tR) in zip(posiRanges, triaRanges) ]
    posiStarts = [ i[0] for i in posiRanges ]
    posiEnds   = [ i[1] for i in posiRanges ]
    triaStarts = [ i[0] for i in triaRanges ]
    triaEnds   = [ i[1] for i in triaRanges ]

    # compute forces for each object
    if 'forces' in outConf:
        if outConf["forces"]:
            ff = [ force(d['forces']) for d in data ]
            [fx, fy, fz] = unzip(ff)
            lists2file(file2write + '-forces', [posiStarts, posiEnds, fx, fy, fz])

    # compute sum of all forces
    if 'total force' in outConf:
        if outConf["total force"]:
            ff = [ force(d['forces']) for d in data ]
            [fx, fy, fz] = unzip(ff)
            f = open(file2write + '-total-force', 'w')
            f.write("{0} {1} {2}".format(sum(fx), sum(fy), sum(fz)))
            f.close()

    # compute sum of all torques
    if 'total torque' in outConf:
        if outConf["total torque"]:
            ff = [torque((0, 0, 0), d['positions'], d['forces']) for d in data]
            [fx, fy, fz] = unzip(ff)
            f = open(file2write + '-total-torque', 'w')
            f.write("{0} {1} {2}".format(sum(fx), sum(fy), sum(fz)))
            f.close()


    # compute torques for each object (TODO: does not work for flagella, there is no origin, yet)
    if 'torques' in outConf:
        if outConf["torques"]:
            # elements of 'objects', which describe flagella, does not contain the attribute 'position'!
            ff = [ torque(o['position'], d['positions'], d['forces']) for (o,d) in zip(objects, data) ]
            [fx, fy, fz] = unzip(ff)
            lists2file(file2write + '-torques', [posiStarts, posiEnds, fx, fy, fz])

    if 'flowfield' in outConf:
        if outConf['flowfield']:
            plotFlowfield(file2write + '-flowfield',
                          concatData([ d['positions'] for d in data ]),
                          concatData([ d['forces'] for d in data ]))
