"""
helper functions for performing surface integration

theta is the angle from 0 to pi
phi   is the angle from 0 to 2*pi
"""
from shapes import spheroidZ
from math import pi,sin,cos,acos,asin,atan2,sqrt


class Surface(object):
    """
    a surface is a list of normal vectors of surfaces and a list of the
    surface's positions.
    """
    def __init__(self, positions, normals):
        self.positions = positions
        self.normals = normals
        self.number = len(positions)

def sphereSurface(radius, nTheta=30):
    from numpy import zeros, array
    grid = surfaceGrid(nTheta)
    num = len(grid)
    positions = zeros((num, 3))
    normals = zeros((num, 3))
    for (c,(t,p)) in zip(range(num), grid):
        (x,y,z) = cartesianFromSphere(radius, t, p)
        positions[c, 0] = x
        positions[c, 1] = y
        positions[c, 2] = z
        dA = radius * radius * sin(t) * grid[(t, p)][0] * grid[(t, p)][1]
        n = array((x, y, z)) / radius * dA
        normals[c, 0] = n[0]
        normals[c, 1] = n[1]
        normals[c, 2] = n[2]
    return Surface(positions, normals)



def xFromSphere(r,theta,phi):
    return r*sin(theta)*cos(phi)
def yFromSphere(r,theta,phi):
    return r*sin(phi)*sin(theta)
def zFromSphere(r,theta,phi):
    return r*cos(theta)
def cartesianFromSphere(r,theta,phi):
    return ( xFromSphere(r,theta,phi)
           , yFromSphere(r,theta,phi)
           , zFromSphere(r,theta,phi))

def cartesianToSphere(x,y,z):
    r = sqrt(x*x + y*y + z*z)
    p = atan2(y,x)
    t = acos(z/r)
    return ( r
           , t
           , p )

def thetaDivision(n):
    return [ pi/n*(i + 0.5) for i in range(n) ]

def thetaDivisionVariation(n):
    return [ pi/n*(i+1) for i in range(n) ]


def phiDivision(n):
    return [ 2*pi/n*(i + 0.5) for i in range(n) ]

def surfaceGridNew(nTheta):
    """
    more homogeneous grid on the surface than with thetaDivision
    and phiDivision. But: dPhi is variable!
    Therefore return a dict with values as keys and differences as values.
    """
    def phisNew(t):
        res = nTheta*(sin(pi-t)+1)+2
        return int(res)

    res = {}
    for (t,counter) in  zip(thetaDivision(nTheta),range(nTheta)):
        for p in phiDivision(phisNew(t)):
            res[(t,p)] = (pi/nTheta,2*pi/phisNew(t))
    return res

def surfaceGrid(nTheta):
    """
    more homogeneous grid on the surface than with thetaDivision
    and phiDivision. But: dPhi is variable!
    Therefore return a dict with values as keys and differences as values.
    """
    def phis(n):
        res = 0
        if n < nTheta/2:
            res = 4*n + 2
        else:
            res = (-4)*(n-nTheta) - 2
        return res

    res = {}
    for (t,counter) in  zip(thetaDivision(nTheta),range(nTheta)):
        for p in phiDivision(phis(counter)):
            res[(t,p)] = (pi/nTheta,2*pi/phis(counter))
    return res

def grid2normal(grid, radius):
    """
    given a grid, produced by e.g. surfaceGrid and a radius.
    return dictionary containing cartesian coordinates and normal vectors
    with the length being the area of the surface elements.
    """
    from numpy import zeros, array
    num = len(grid)
    xs = zeros((num))
    ys = zeros((num))
    zs = zeros((num))
    nx = zeros((num))
    ny = zeros((num))
    nz = zeros((num))
    for (c,(t,p)) in zip(range(num), grid):
        (x,y,z) = cartesianFromSphere(radius, t, p)
        xs[c] = x
        ys[c] = y
        zs[c] = z
        dA = radius * radius * sin(t) * grid[(t, p)][0] * grid[(t, p)][1]
        n = array((x, y, z)) / radius * dA
        nx[c] = n[0]
        ny[c] = n[1]
        nz[c] = n[2]
    return {'x': xs, 'y': ys, 'z': zs, 'nx': nx, 'ny': ny, 'nz': nz}

def grid2numpySphere(grid, radius):
    """
    grid is obtained by, e.g. surfaceGrid function.
    """
    from numpy import zeros
    num = len(grid)
    thetas  = zeros([num])
    phis    = zeros([num])
    dthetas = zeros([num])
    dphis   = zeros([num])
    xs = zeros([num])
    ys = zeros([num])
    zs = zeros([num])
    for (c,(t,p)) in zip(range(num), grid):
        (x,y,z) = cartesianFromSphere(radius, t, p)
        xs[c] = x
        ys[c] = y
        zs[c] = z
        thetas[c] = t
        phis[c] = p
        dthetas[c] = grid[(t,p)][0]
        dphis[c] = grid[(t,p)][1]
    return (thetas, phis, dthetas, dphis, xs, ys, zs)

def grid2numpyEllipsoid(grid, shape):
    from numpy import zeros
    num = len(grid)
    thetas  = zeros([num])
    phis    = zeros([num])
    dthetas = zeros([num])
    dphis   = zeros([num])
    xs = zeros([num])
    ys = zeros([num])
    zs = zeros([num])
    for (c,(t,p)) in zip(range(num), grid):
        (x,y,z) = cartesianFromSphere(shape(t,p), t, p)
        xs[c] = x
        ys[c] = y
        zs[c] = z
        thetas[c] = t
        phis[c] = p
        dthetas[c] = grid[(t,p)][0]
        dphis[c] = grid[(t,p)][1]
    return (thetas, phis, dthetas, dphis, xs, ys, zs)
    

def integrateSphere(f, N=10, radius=1, origin=(0,0,0)):
    """
    surface integration with a homogeneous set of surface points
    """
    res = 0
    grid = surfaceGrid(N)
    for (t,p) in grid:
        (x,y,z) = cartesianFromSphere(radius,t,p)
        (dt,dp) = grid[(t,p)]
        res += f(x+origin[0], y+origin[1], z+origin[2])*radius*radius*sin(t)*dp*dt
    return res

def integrate(f,N=10,shape=lambda t,p: 1,origin=(0,0,0)):
    """
    surfaceIntegration with a homogeneous set of surface points
    """
    res = 0
    grid = surfaceGrid(N)
    for (t,p) in grid:
        r = shape(t,p)
        (x,y,z) = cartesianFromSphere(r,t,p)
        (dt,dp) = grid[(t,p)]
        res += f(origin[0]+x,origin[1]+y,origin[2]+z)*r*r*sin(t)*dp*dt
    return res



def integrateSpheroid(f, N=10, a=0.8, c=1.2, origin=(0,0,0)):
    """
    surface integration over a spheroid.
    The surface element has to be adjusted.
    In this integration method, only oblats and prolats
    in z-direction are considered
    """
    res = 0
    r = spheroidZ(a,c)
    drdt = lambda t,p: r(t,p)**3/((-1)*a*a*c*c)*sin(t)*cos(t)*(c**2 - a**2)
    grid = surfaceGrid(N)
    for (t,p) in grid:
        prefactor_dA = r(t,p)*sin(t)*sqrt(r(t,p)**2 + drdt(t,p)**2)
        (dt,dp) = grid[(t,p)]
        dA = prefactor_dA*dt*dp
        (x,y,z) = cartesianFromSphere(r(t,p),t,p)
        res += f(x+origin[0], y+origin[1], z+origin[2])*dA
    return res

def integrateShape(f, r, drdt, drdp, N=10):
    """
    if we want to integrate over an arbitrary shape, given as
    a function r = r(t,p), we can calculate surface elements by
    knowing derivatives dr/dt and dr/dp, both function relativ
    to theta and phi.
    """
    res = 0
    grid = surfaceGrid(N)
    for (t,p) in grid:
        prefactor_dA = sqrt(r(t,p)**2*(r(t,p)**2*sin(t)**2 + drdp(t,p)**2 + drdt(t,p)**2*sin(t)**2))
        (dt,dp) = grid[(t,p)]
        dA = prefactor_dA*dt*dp
        (x,y,z) = cartesianFromSphere(r(t,p), t, p)
        res += f(x,y,z)*dA
    return res

def integrateArbitraryShape(f, r, N=10):
    """
    if we want to integrate over an arbitrary shape, given as
    a function r = r(t,p), we can calculate surface elements by
    knowing derivatives dr/dt and dr/dp, both function are
    derived numerically.
    """
    res = 0
    grid = surfaceGrid(N)
    e = 0.001
    for (t,p) in grid:
        drdt = (r(t-e,p) - r(t+e,p))/(2*e)
        drdp = (r(t,p-e) - r(t,p+e))/(2*e)
        prefactor_dA = sqrt(r(t,p)**2*(r(t,p)**2*sin(t)**2 + drdp**2 + drdt**2*sin(t)**2))
        (dt,dp) = grid[(t,p)]
        dA = prefactor_dA*dt*dp
        (x,y,z) = cartesianFromSphere(r(t,p), t, p)
        res += f(x,y,z)*dA
    return res
