"""
ben's calculations save some data: geometry and fourier modes of the trajectory.
with this module, we can reconstruct everything in cartesian coordinates.

Warning: its 2d!
add one dimension for being useful to FMBEM!

Then plotting with blender and simulations with FMBEM are wanted.
"""

from misc import takeOneOf

def readNumber(filename):
    f = open(filename, 'r')
    res = float(f.read())
    f.close()
    return res

def readArray(filename):
    f = open(filename, 'r')
    res = []
    h = f.read().split()
    res = [float(i) for i in h]
    f.close()
    return res

def readData(folder='.'):
    """
    read ben's matlab data
    """
    from os import chdir, getcwd
    currentFolder = getcwd()
    chdir(folder)
    A = readNumber('A')
    B = readNumber('B')
    L = readNumber('L')
    r1 = readNumber('r1')
    r2 = readNumber('r2')
    psi0   = readArray('psi0')
    psi1re = readArray('psi1re')
    psi1im = readArray('psi1im')
    psi2re = readArray('psi2re')
    psi2im = readArray('psi2im')
    psi3re = readArray('psi3re')
    psi3im = readArray('psi3im')
    chdir(currentFolder)
    from numpy import array
    return {'A': A,
            'B': B,
            'L': L,
            'r1': r1,
            'r2': r2,
            'psi0': array(psi0),
            'psi1': {'re': array(psi1re), 'im': array(psi1im) },
            'psi2': {'re': array(psi2re), 'im': array(psi2im) },
            'psi3': {'re': array(psi3re), 'im': array(psi3im) } }

def data2rftInit(data):
    from numpy import sqrt
    dx = data['r2']
    dy = data['r1']
    body = (dx, dy)
    bx = data['A']
    by = data['B'] # sqrt(1 - (bx / dx) ** 2) * dy
    L = data['L']
    class Ret(object):
        def __init__(self):
            self.body = body
            self.bx = bx
            self.by = by
            self.L = L
    return Ret()


def data2rftPsis(data, omega, time, amplitude=1):
    psiR = flagellaRightPsi(0, omega, time,
                            data['psi0'], data['psi1'],
                            data['psi2'], data['psi3'],
                            amplitude=amplitude)
    psiL = flagellaLeftPsi(0, omega, time,
                            data['psi0'], data['psi1'],
                            data['psi2'], data['psi3'],
                            amplitude=amplitude)
    dpsiR = flagellaRightdPsi(0, omega, time,
                              data['psi0'], data['psi1'],
                              data['psi2'], data['psi3'],
                              amplitude=amplitude)
    dpsiL = flagellaLeftdPsi(0, omega, time,
                             data['psi0'], data['psi1'],
                             data['psi2'], data['psi3'],
                             amplitude=amplitude)
    class Ret(object):
        def __init__(self):
            self.psiL = psiL
            self.psiR = psiR
            self.dpsiL = dpsiL
            self.dpsiR = dpsiR
    return Ret()



def flagellaLeftPsi(alpha, omega, time, psi0, psi1, psi2, psi3, amplitude=1):
    """
    alpha is a starting angle, the orientation of the flagellum.
    psi's are the fourier modes as numpy arrays.
    """
    return flagellaLeftPsiFromPhi(alpha, omega * time, psi0, psi1, psi2, psi3, amplitude=amplitude)

def flagellaLeftPsiFromPhi(alpha, phi, psi0, psi1, psi2, psi3, amplitude=1):
    from numpy import pi, sin, cos
    phi_0 = alpha + pi - psi0
    phiL  = -psi1['re']*cos(phi) - psi1['im']*sin(phi)
    phiL += -psi2['re']*cos(2*phi) - psi2['im']*sin(2*phi)
    phiL += -psi3['re']*cos(3*phi) - psi3['im']*sin(3*phi)
    return phi_0 + amplitude * phiL

def flagellaLeftdPsi(alpha, omega, time, psi0, psi1, psi2, psi3, amplitude=1):
    """
    alpha is a starting angle, the orientation of the flagellum.
    psi's are the fourier modes as numpy arrays.
    """
    from numpy import pi, sin, cos
    dphiL  = omega*(psi1['re']*sin(omega*time) - psi1['im']*cos(omega*time))
    dphiL += 2*omega*(psi2['re']*sin(2*omega*time) - psi2['im']*cos(2*omega*time))
    dphiL += 3*omega*(psi3['re']*sin(3*omega*time) - psi3['im']*cos(3*omega*time))
    return dphiL * amplitude

def flagellaRightPsi(alpha, omega, time, psi0, psi1, psi2, psi3, amplitude=1):
    """
    alpha is a starting angle, the orientation of the flagellum..
    psi's are the fourier modes as numpy arrays.
    """
    return flagellaRightPsiFromPhi(alpha, omega * time, psi0, psi1, psi2, psi3, amplitude=amplitude)

def flagellaRightPsiFromPhi(alpha, phi, psi0, psi1, psi2, psi3, amplitude=1):
    from numpy import pi, sin, cos
    phi_0 = alpha + psi0
    phiR  = psi1['re']*cos(phi) + psi1['im']*sin(phi)
    phiR += psi2['re']*cos(2*phi) + psi2['im']*sin(2*phi)
    phiR += psi3['re']*cos(3*phi) + psi3['im']*sin(3*phi)
    return phi_0 + amplitude * phiR

def flagellaRightdPsi(alpha, omega, time, psi0, psi1, psi2, psi3, amplitude=1):
    """
    alpha is a starting angle, the orientation of the flagellum..
    psi's are the fourier modes as numpy arrays.
    """
    from numpy import pi, sin, cos
    dphiR  = (-psi1['re']*sin(omega*time) + psi1['im']*cos(omega*time))*omega
    dphiR += (-psi2['re']*sin(2*omega*time) + psi2['im']*cos(2*omega*time))*2*omega
    dphiR += (-psi3['re']*sin(3*omega*time) + psi3['im']*cos(3*omega*time))*2*omega
    return dphiR * amplitude

def leftPsi2cartesian(A, B, alpha, psi, ds):
    """
    A and B describe the starting position with respect to the cell body,
    alpha is some initial angle, the orientation of the flagellum.,
    take psi values computed by flagella{Left,Right}Psi
    and distance of parametrization.
    return flagella in cartesian coordinates.
    """
    from numpy import sin, cos
    from scipy.integrate import cumtrapz
    x = -A*cos(alpha) - B*sin(alpha) + cumtrapz(cos(psi), dx=ds)
    y = -A*sin(alpha) + B*cos(alpha) + cumtrapz(sin(psi), dx=ds)
    return (x,y)

def rightPsi2cartesian(A, B, alpha, psi, ds):
    """
    A and B describe the starting position with respect to the cell body,
    alpha is some initial angle, the orientation of the flagellum.,
    take psi values computed by flagella{Left,Right}Psi
    and distance of parametrization.
    return flagella in cartesian coordinates.
    """
    from numpy import sin, cos
    from scipy.integrate import cumtrapz
    x = A*cos(alpha) - B*sin(alpha) + cumtrapz(cos(psi), dx=ds)
    y = A*sin(alpha) + B*cos(alpha) + cumtrapz(sin(psi), dx=ds)
    return (x,y)

def flagellaLeftCoordinates(alpha, omega, ds, data, time):
    psiL = flagellaLeftPsi(alpha, omega, time, data['psi0'], data['psi1'], data['psi2'], data['psi3'])
    coordsL = leftPsi2cartesian(data['A'], data['B'], alpha, psiL, ds)
    return coordsL

def flagellaRightCoordinates(alpha, omega, ds, data, time):
    psiR = flagellaRightPsi(alpha, omega, time, data['psi0'], data['psi1'], data['psi2'], data['psi3'])
    coordsR = rightPsi2cartesian(data['A'], data['B'], alpha, psiR, ds)
    return coordsR

def cartesianTrajectories(alpha, omega, ds, data, time):
    coordsL = flagellaLeftCoordinates(alpha, omega, ds, data, time)
    coordsR = flagellaRightCoordinates(alpha, omega, ds, data, time)
    return { 'left': coordsL, 'right': coordsR }

def threeD(point):
    """ the point lie in the xy-plane at z = 0 """
    return (point[0], point[1], 0)

def constructFMBEMobjects(data, alpha, period, time, dt):
    """
    data is the data object containing everything what Ben provide
    from his force resistive theory calculations.

    it return objects (dictionaries) that can be indirectly used as input
    for the FMBEM calculations, i.e. as objects inside a python script that produces
    the FMBEM input.dat file and the data analyzing remembery.py file.
    """
    from math import pi
    omega = 2*pi/period
    numberOfPoints = len(data['psi0'])
    ds = data['L']/numberOfPoints
    ## flagella trajectories contain a thousend points
    #  that's too much for the triangulation process
    #  for test purposes take only 100?
    # flagella trajectories
    h1 = cartesianTrajectories(alpha, omega, ds, data, time)
    coordsL = h1['left']
    coordsR = h1['right']
    # flagella future trajectories
    h2 = cartesianTrajectories(alpha, omega, ds, data, time + dt)
    fCoordsL = h2['left']
    fCoordsR = h2['right']
    # flagella interface:
    radius = 12.5*ds  # data from Lenaghan, Chen, Zhang, 2013
    grid = 9 # azimuth grid corresponds to the number of microtubuli
    flagellumL = { 'positions': takeOneOf(30, [threeD(p) for p in zip(coordsL[0], coordsL[1]) ]),
                  'future positions': takeOneOf(30, [threeD(p) for p in zip(fCoordsL[0], fCoordsL[1]) ]),
                  'radius': radius,
                  'azimuth grid': grid }
    flagellumR = { 'positions': takeOneOf(30, [threeD(p) for p in zip(coordsR[0], coordsR[1]) ]),
                  'future positions': takeOneOf(30, [threeD(p) for p in zip(fCoordsR[0], fCoordsR[1]) ]),
                  'radius': radius,
                  'azimuth grid': grid }
    # cell body interface:
    body = { 'position': (0,0,0),
             'lengths': (data['r1'], data['r2'], data['r2']),
             'axe1': (1,0,0),
             'axe2': (0,1,0),
             'grid': 30,
             'velocity': (0,0,0),
             'angular': (0,0,0) }
    return { 'body': body,
             'flagella': [flagellumL, flagellumR] }

def constructFMBEMobjectsFromDataFolder(foldername, alpha, period, time, dt):
    """ see constructFMBEMobjects for more details """
    data = readData(foldername)
    return constructFMBEMobjects(data, alpha, period, time, dt)

def blabla(period):
    from math import pi
    alpha = 0
    omega = 2*pi/period
    data = readData()
    numberOfPoints = len(data['psi0'])
    ds = data['L']/numberOfPoints
    xxL = []
    xxR = []
    yyL = []
    yyR = []
    for t in [i for i in range(period)]:
        h = cartesianTrajectories(alpha, omega, ds, data, t)
        xxL.append(h['left'][0])
        xxR.append(h['right'][0])
        yyL.append(h['left'][1])
        yyR.append(h['right'][1])
    return (data['r1'], data['r2'], xxL, xxR, yyL, yyR)
