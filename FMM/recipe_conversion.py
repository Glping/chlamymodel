import numpy as np

def orientation(origin, points):
    origin = np.array(origin)
    [p1, p2, p3] = [ np.array(points[p]) - origin for p in [0,1,2] ]
    v1 = p1 - p2
    v2 = p3 - p2
    normal = np.cross(v1, v2)
    normal = normal/np.linalg.norm(normal)
    orientation = p1 + p2 + p3
    orientation = orientation/np.linalg.norm(orientation)
    #if norm(orientation + normal) < sqrt(2):
    if np.dot(orientation, normal) < 0:
        res = 0
    else:
        res = 1
    return res

def orientate(origin, points, triangle):
    """
    test the orientation of points and correct it, eventually.
    points is a list of three points, that pertain to the triangle
    indices, respectively.
    """
    if orientation(origin, points) == 0:
        triangle = (triangle[0], triangle[2], triangle[1])
    return triangle

def orientateTriangulation(origin, points, triangulation):
    triangulationH = []
    for tri in triangulation:
        triH = orientate(origin, [points[tri[0]], points[tri[1]], points[tri[2]]], tri)
        triangulationH.append(triH)
    return triangulationH

def prepareFlagella(radius=1,
                    points1=[],
                    points2=[],
                    nTheta=6,
                    dt=1):
    """
    a flagella is represented as a list of points.
    given two such flagella allows us to compute the velocity
    at each point.
    """
    import triangulate
    points1 = list(map(lambda c: np.array(c), points1))
    points2 = list(map(lambda c: np.array(c), points2))
    flagella = triangulate.Flagella(radius, points1, nTheta)
    coordinates = []
    velocities = []
    triangulation = []
    (lineElements, circleElements, coords) = flagella.surfaceParametrization.shape
    for le in range(lineElements):
        for ce in range(circleElements):

            coordinates.append(flagella.surfaceParametrization[le, ce])
            velocities.append(0.5 / dt * (points2[le] - points1[le] + points2[le+1] - points1[le+1]))

    triangulation = flagella.triangulate()

    ### take into account the first and last point
    coordinates.append(points1[0])
    coordinates.append(points1[-1])
    velocities.append((points2[0] - points1[0]) / dt)
    velocities.append((points2[-1] - points1[-1]) / dt)

    # the first point connects to the first circleElements of coordinates
    # be careful with orientation
    point1Index = lineElements * circleElements
    someIndices = list(range(circleElements))
    for (i2, i3) in zip(someIndices[:-1], someIndices[1:]):
        triangulation.append((point1Index, i2, i3))
    triangulation.append((point1Index, someIndices[-1], someIndices[0]))

    # the last point connects to the last circleElements of coordinates - 2
    point2Index = point1Index + 1
    someIndices = [ i + (lineElements - 1) * circleElements for i in range(circleElements) ]
    for (i2, i3) in zip(someIndices[:-1], someIndices[1:]):
        triangulation.append((point2Index, i3, i2))
    triangulation.append((point2Index, someIndices[0], someIndices[-1]))

    return (coordinates, velocities, triangulation)


def prepareWorm(radius=lambda r: r[0],
                points1=None,
                points2=None,
                nTheta=6,
                dt=1):

    (c, v, t) = prepareFlagella(radius=radius,
                                points1=points1,
                                points2=points2,
                                nTheta=nTheta,
                                dt=dt)
    for i in range(len(c)):
        c[i] = radius(c[i]) * c[i]
        v[i] = radius(c[i]) * v[i]

    return (c, v, t)


def prepareSphere(position, radius, velocity, angular, gridLevel):
    """ a moving sphere """
    import triangulate
    import rotate
    (coordinates, triangulation) = triangulate.sphere(position, radius, gridLevel)
    position = np.array(position)
    coordinates = [np.array(i) + position for i in coordinates]
    velocities = []
    for c in coordinates:
        # vector to be rotated
        toBeRotated = np.array(c) - position
        # rotated vector
        #rotated = rotate.rotateVector(toBeRotated, norm(angular), angular)
        veloFromAng = np.cross(toBeRotated, angular)
        # velocity of the surface element
        velocities.append(veloFromAng + velocity)

    triangulation = orientateTriangulation(position, coordinates, triangulation)

    return (coordinates, velocities, triangulation)

def prepareEllipsoid(position=(0, 0, 0),
                     lengths=(1, 1, 1),
                     axe1=(1, 0, 0),
                     axe2=(0, 1, 0),
                     velocity=(0, 0, 0),
                     angular=(0, 0, 0),
                     grid=30):
    """ a moving sphere """
    import sphere
    ellipsoid = sphere.Ellipsoid(position=position, lengths=lengths, axe1=axe1, axe2=axe2, grid=grid)
    return prepareConvexShape(ellipsoid, velocity, angular)


def prepareBacterialHead(length=1, radius=1, grid=30):
    """ model the head as a convex shape, elongated in x-direction """
    import rotate
    coordinates = prepareEllipsoid(lengths=(radius, radius, radius),
                                   grid=grid)[0]
    points = []
    for (i, c) in enumerate(coordinates):
        if c[0] > 0.05 * radius:
            points.append((c[0] + length / 2 - 0.05 / 2, c[1], c[2]))
        elif c[0] < -0.05 * radius:
            points.append((c[0] - length / 2 + 0.05 / 2, c[1], c[2]))

    rot = rotate.axis2matrix([2 * np.pi / grid, 0, 0])
    grid_l = round(grid * length / (2 * np.pi * radius))
    dl = length / grid_l
    height = lambda x: (radius + radius / 100) - (radius / 100) / (length / 2) ** 2 * x ** 2
    vect = [0, 1, 0]
    for i in range(grid):
        vect = np.dot(rot, vect)
        for j in range(grid_l):
            (x, y, z) = vect
            x = (j + 0.5) * dl - length / 2
            points.append([x, height(x) * y, height(x) * z])

    # for (i, (x, y, z)) in enumerate(points):
    #     points[i] = (x, radius * y, radius * z)

    # import matplotlib.pylab as plt
    # plt.plot([p[0] for p in points], [p[1] for p in points], color=(1, 1, 1), marker='o')
    # plt.show()

    class BactHead(object):
        def __init__(self):
            self.grid = (0, 0, 0, 0,
                         [p[0] for p in points],
                         [p[1] for p in points],
                         [p[2] for p in points])
            self.position = (0, 0, 0)
    return prepareConvexShape(BactHead(), (0, 0, 0), (0, 0, 0))

def prepareConvexShape(shape, velocity, angular):
    """
    a moving sphere
    """
    from scipy.spatial import ConvexHull

    velocity = np.array(velocity)

    # coordinates
    coordinates = []
    for coord in zip(shape.grid[4], shape.grid[5], shape.grid[6]):
        coordinates.append(coord)

    # velocities
    origin = np.array(shape.position)
    velocities = []
    for v in coordinates:

        # vector to be rotated
        toBeRotated = np.array(v) - origin
        # rotated vector
        #rotated = rotate.rotateVector(toBeRotated, norm(angular), angular)
        veloFromAng = -np.cross(toBeRotated, angular)
        # velocity of the surface element
        velocities.append(veloFromAng + velocity)

    # triangulation
    triangulation = orientateTriangulation(origin, coordinates, ConvexHull(coordinates).simplices)

    return (coordinates, velocities, triangulation)


def prepareFlowingPlane(p0=(-1, -1, -1),
                        p1=(1, 1, 1),
                        p2=(0, 0, 1),
                        grid1=3,
                        grid2=3,
                        centers=[],
                        velocity_field=lambda r: (0, 0, 0)):
    """
    create rectangular mesh from p1 to p2 with some mesh density and centers
    with increased density.
    """
    p0 = np.array(p0)
    dp1 = np.array(p1) - np.array(p0)
    dp2 = np.array(p2) - np.array(p0)
    c = np.zeros(((grid1 + 1) * (grid2 + 1), 3))
    v = np.zeros(((grid1 + 1) * (grid2 + 1), 3))
    fI = lambda k, l: k * (grid2 + 1) + l
    for  i in range(grid1 + 1):
        alpha = i / float(grid1)
        for j in range(grid2 + 1):
            beta = j / float(grid2)

            c[fI(i, j)] = p0 + alpha * dp1 + beta * dp2
            v[fI(i, j)] = velocity_field(c[fI(i, j)])

    t = []
    for i in range(grid1):
        for j in range(grid2):
            t.append([fI(i, j), fI(i+1, j), fI(i+1, j+1)])
            t.append([fI(i, j), fI(i+1, j+1), fI(i, j+1)])

    dmax = 20
    alpha = lambda d: 0.7 * (d / dmax) + 0.3
    for center in centers:
        for i in range(grid1 + 1):
            for j in range(grid2 + 1):
                if i == 0 or i == grid1:
                    continue
                if j == 0 or j == grid2:
                    continue
                dij = c[fI(i, j)] - center
                ldij = np.linalg.norm(dij)
                if ldij < dmax:
                    c[fI(i, j)] = alpha(ldij) * dij + center

    return (c, v, t)

def preparePlane(p0=(-1, -1, -1),
                 p1=(1, 1, 1),
                 p2=(0, 0, 1),
                 width=1,
                 grid1=3,
                 grid2=3,
                 centers=[]):
    """
    create rectangular mesh from p1 to p2 with some mesh density and centers
    with increased density.
    """
    p0 = np.array(p0)
    dp1 = np.array(p1) - np.array(p0)
    dp2 = np.array(p2) - np.array(p0)
    dp3 = np.cross(dp1, dp2)
    dp3 = dp3 / np.linalg.norm(dp3) * width
    origin = p0 + 0.5 * (dp1 + dp2 + dp3)
    c = np.zeros((2 * (grid1 + 1) * (grid2 + 1), 3))
    v = np.zeros((2 * (grid1 + 1) * (grid2 + 1), 3))
    fI1 = lambda k, l: k * (grid2 + 1) + l
    fI2 = lambda k, l: k * (grid2 + 1) + l + (grid1 + 1) * (grid2 + 1)
    for  i in range(grid1 + 1):
        alpha = i / float(grid1)
        for j in range(grid2 + 1):
            beta = j / float(grid2)

            c[fI1(i, j)] = p0 + alpha * dp1 + beta * dp2
            c[fI2(i, j)] = p0 + alpha * dp1 + beta * dp2 + dp3
            v[fI1(i, j)] = (0, 0, 0) # velocityfield(c[fI1(i, j)][0],
                           #               c[fI1(i, j)][1],
                           #               c[fI1(i, j)][2])
            v[fI2(i, j)] = (0, 0, 0) # velocityfield(c[fI2(i, j)][0],
                           #               c[fI2(i, j)][1],
                           #               c[fI2(i, j)][2])
    t = []
    for i in range(grid1):
        for j in range(grid2):
            t.append([fI1(i, j), fI1(i+1, j), fI1(i+1, j+1)])
            t.append([fI2(i, j), fI2(i+1, j + 1), fI2(i+1, j)])
            t.append([fI1(i, j), fI1(i+1, j+1), fI1(i, j+1)])
            t.append([fI2(i, j), fI2(i, j+1), fI2(i+1, j+1)])
    for i in range(grid1):
        t.append([fI1(i, grid2), fI2(i + 1, grid2), fI1(i + 1, grid2)])
        t.append([fI1(i, grid2), fI2(i, grid2), fI2(i + 1, grid2)])
        t.append([fI1(i + 1, 0), fI1(i, 0), fI2(i, 0)])
        t.append([fI1(i + 1, 0), fI2(i + 1, 0), fI2(i, 0)])
    for j in range(grid2):
        t.append([fI1(grid1, j + 1), fI1(grid1, j), fI2(grid1, j)])
        t.append([fI1(grid1, j + 1), fI2(grid1, j), fI2(grid1, j + 1)])
        t.append([fI1(0, j), fI2(0, j + 1), fI1(0, j + 1)])
        t.append([fI1(0, j), fI2(0, j), fI2(0, j + 1)])


    dmax = 20
    alpha = lambda d: 0.7 * (d / dmax) + 0.3
    for center in centers:
        for i in range(grid1 + 1):
            for j in range(grid2 + 1):
                if i == 0 or i == grid1:
                    continue
                if j == 0 or j == grid2:
                    continue
                dij = c[fI1(i, j)] - center
                ldij = np.linalg.norm(dij)
                if ldij < dmax:
                    c[fI1(i, j)] = alpha(ldij) * dij + center
                dij = c[fI2(i, j)] - center
                ldij = np.linalg.norm(dij)
                if ldij < dmax:
                    c[fI2(i, j)] = alpha(ldij) * dij + center
    t = orientateTriangulation(origin, c, t)
    return (c, v, t)

def preparePlane_old(p0=(-1, -1, -1),
                 p1=(1, 1, 1),
                 p2=(0, 0, 1),
                 grid1=3,
                 grid2=3,
                 centers=[]):
    """
    create rectangular mesh from p1 to p2 with some mesh density and centers
    with increased density.
    """
    p0 = np.array(p0)
    dp1 = np.array(p1) - np.array(p0)
    dp2 = np.array(p2) - np.array(p0)
    c = np.zeros(((grid1 + 1) * (grid2 + 1), 3))
    v = np.zeros(((grid1 + 1) * (grid2 + 1), 3))
    fI = lambda k, l: k * (grid2 + 1) + l
    for  i in range(grid1 + 1):
        alpha = i / float(grid1)
        for j in range(grid2 + 1):
            beta = j / float(grid2)

            c[fI(i, j)] = p0 + alpha * dp1 + beta * dp2
            v[fI(i, j)] = (0, 0, 0) #velocityfield(c[fI(i, j)][0],
                          #              c[fI(i, j)][1],
                          #              c[fI(i, j)][2])
    t = []
    for i in range(grid1):
        for j in range(grid2):
            t.append([fI(i, j), fI(i+1, j), fI(i+1, j+1)])
            t.append([fI(i, j), fI(i+1, j+1), fI(i, j+1)])

    dmax = 20
    alpha = lambda d: 0.7 * (d / dmax) + 0.3
    for center in centers:
        for i in range(grid1 + 1):
            for j in range(grid2 + 1):
                if i == 0 or i == grid1:
                    continue
                if j == 0 or j == grid2:
                    continue
                dij = c[fI(i, j)] - center
                ldij = np.linalg.norm(dij)
                if ldij < dmax:
                    c[fI(i, j)] = alpha(ldij) * dij + center

    return (c, v, t)

def prepareRectangle(p0=(0, 0, 0),
                     p1=(0, 1, 0),
                     p2=(0, 0, 1),
                     p3=(1, 0, 0),
                     grid1=3,
                     grid2=3,
                     grid3=3,
                     velocityfield=lambda x, y, z: (0, 0, 0)):
    p0 = np.array(p0)
    dp1 = np.array(p1) - p0
    dp2 = np.array(p2) - p0
    dp3 = np.array(p3) - p0
    origin = p0 + 0.5 * (dp1 + dp2 + dp3)

    coords = []
    for i in range(grid1 + 1):
        alpha = i / float(grid1)
        for j in range(grid2 + 1):
            beta = j / float(grid2)
            coords.append(p0 + alpha * dp1 + beta * dp2)
            coords.append(p0 + alpha * dp1 + beta * dp2 + dp3)
    for i in range(grid1 + 1):
        alpha = i / float(grid1)
        for j in range(grid3):
            beta = (j + 1) / float(grid3)
            coords.append(p0 + alpha * dp1 + beta * dp3)
            coords.append(p0 + alpha * dp1 + beta * dp3 + dp2)
    for i in range(grid3):
        alpha = (i + 1) / float(grid3)
        for j in range(grid2):
            beta = (j + 1) / float(grid2)
            coords.append(p0 + alpha * dp3 + beta * dp2)
            coords.append(p0 + alpha * dp3 + beta * dp2 + dp1)

    v = []
    for c in coords:
        v.append(velocityfield(c[0], c[1], c[2]))
    from scipy.spatial import Delaunay
    d = Delaunay(coords)
    t = orientateTriangulation(origin, coords, d.convex_hull)

    return (coords, v, t)


def meshProductionRule2Mesh(rule):
    """
    rule is a dictionary and represents an ellipse or a flagellum
    (and later a plane, too),
    returns a dictionary of lists of triples, representing coordinates,
    velocities and triangulation.
    """

    if rule['type'] == 'ellipsoid':
        (c, v, t) = prepareEllipsoid(position=(0, 0, 0), #rule['position'],
                                     lengths=rule['lengths'],
                                     axe1=rule['axe1'],
                                     axe2=rule['axe2'],
                                     grid=rule['grid'])

    elif rule['type'] == 'bacterial head':
        (c, v, t) = prepareBacterialHead(length=rule['length'],
                                         radius=rule['radius'],
                                         grid=rule['grid'])

    elif rule['type'] == 'worm':
        (c, v, t) = prepareWorm(radius=rule['radius'],
                                points1=rule['positions'],
                                points2=rule['future positions'],
                                nTheta=rule['azimuth grid'],
                                dt=rule['dt'])

    elif rule['type'] == 'flagellum':
        (c, v, t) = prepareFlagella(radius=rule['radius'],
                                    points1=rule['positions'],
                                    points2=rule['future positions'],
                                    nTheta=rule['azimuth grid'],
                                    dt=rule['dt'])

    elif rule['type'] == 'flatplane':
        (c, v, t) = preparePlane_old(p0=rule['p0'],
                                     p1=rule['p1'],
                                     p2=rule['p2'],
                                     grid1=rule['grid1'],
                                     grid2=rule['grid2'],
                                     centers=rule['centers'])

    elif rule['type'] == 'flowing plane':
        (c, v, t) = prepareFlowingPlane(p0=rule['p0'],
                                        p1=rule['p1'],
                                        p2=rule['p2'],
                                        grid1=rule['grid1'],
                                        grid2=rule['grid2'],
                                        centers=rule['centers'],
                                        velocity_field=rule['velocity_field'])

    elif rule['type'] == 'plane':
        (c, v, t) = preparePlane(p0=rule['p0'],
                                 p1=rule['p1'],
                                 p2=rule['p2'],
                                 width=rule['width'],
                                 grid1=rule['grid1'],
                                 grid2=rule['grid2'],
                                 centers=rule['centers'])

    elif rule['type'] == 'rectangle':
        (c, v, t) = prepareRectangle(p0=rule['p0'],
                                     p1=rule['p1'],
                                     p2=rule['p2'],
                                     p3=rule['p3'],
                                     grid1=rule['grid1'],
                                     grid2=rule['grid2'],
                                     grid3=rule['grid3'],
                                     velocityfield=rule['velocityfield'])

    return {'coordinates': np.array(c),
            'velocities': np.array(v),
            'triangulation': np.array(t)}
