"""
functions for creating shapes, which can be used for integration
in surfaceIntegration. These are function lambda theta,phi: ...
"""
from math import cos, sin, sqrt

def sphere(t,p):
    return 1

def spheroidZ(a,c):
    """
    this function provides the exact spheroid shape
    for oblats (a<c) and prolats(a>c). For a == c it's the sphere shape.
    derived from the spheroid equation with transformation
    to spherical coordinates.
    """
    return lambda t,p: sqrt((a**2*c**2)/(c**2*sin(t)**2 + a**2*cos(t)**2))

def getSpheroidA(lengths,axe1,axe2):
    """
    construct matrix from two eigenvectors that should be orthogonal to each other
    (due to the interpretation of the values for an ellipse) and eigenvalues (which are
    interpreted as length for principle axes). The generated output can be used as
    input for the function spheroid and co.
    """
    from numpy import diag, array, cross, dot, transpose, sqrt
    from numpy.linalg import inv, norm
    L = diag(1/(array(lengths))**2)
    axe3 = cross(axe1,axe2)
    axe2 = cross(axe1,axe3) # if axe1 and axe2 are not orthogonal
    axe1 = array(axe1)/norm(axe1)
    axe2 = array(axe2)/norm(axe2)
    axe3 = array(axe3)/norm(axe3)
    Ph = array([axe1.tolist(),axe2.tolist(),axe3.tolist()])
    P = transpose(Ph)
    PInv = inv(P)
    return dot(dot(P,L),PInv)

def spheroid(A):
    """
    a general notation of a spheroid is x^T*A*x == 1. The eigenvector
    of A determine the orientation of the principle axis and the
    eigenvalues are the squares of the lengths of these axis.
    r was calculated using mathematika.
    """
    r1 = lambda t,p: sin(t)**2*(A[0][0]*cos(p)**2 + (A[0][1] + A[1][0])*sin(p)*cos(p) + A[1][1]*sin(p)**2)
    r2 = lambda t,p: sin(t)*cos(t)*((A[0][2] + A[2][0])*cos(p) + (A[1][2] + A[2][1])*sin(p))
    r3 = lambda t,p: cos(t)**2*A[2][2]
    return lambda t,p: 1/sqrt(r1(t,p) + r2(t,p) + r3(t,p))

def DspheroidDt(A):
    """ derivatves of the shape function, used for integration """
    r1 = lambda t,p: -2*cos(2*t)*((A[0][2] + A[2][0])*cos(p) + (A[1][2] + A[2][1])*sin(p))
    r2 = lambda t,p: (-1)*(A[0][0] + A[1][1] - 2*A[2][2] + (A[0][0] - A[1][1])*cos(2*p) + (A[0][1] + A[1][0])*sin(2*p))*sin(2*t)
    r3 = lambda t,p: 4*(A[2][2]*cos(t)**2 + cos(t)*((A[0][2] + A[2][0])*cos(p) + (A[1][2] + A[2][1])*sin(p))*sin(t))
    r4 = lambda t,p: 4*(A[0][0]*cos(p)**2 + (A[0][1] + A[1][0])*cos(p)*sin(p) + A[1][1]*sin(p)**2)*sin(t)**2
    return lambda t,p: (r1(t,p) + r2(t,p))/sqrt(r3(t,p) + r4(t,p))**3

def DspheroidDp(A):
    """ derivatves of the shape function, used for integration """
    r1 = lambda t,p: sin(t)*((-(A[1][2] + A[2][1]))*cos(p)*cos(t) + (A[0][2] + A[2][0])*cos(t)*sin(p))
    r2 = lambda t,p: ((A[0][1] + A[1][0])*cos(2*p) + (A[1][1] - A[0][0])*sin(2*p))*sin(t)**2
    r3 = lambda t,p: 2*(A[2][2]*cos(t)**2 + cos(t)*((A[0][2] + A[2][0])*cos(p) + (A[1][2] + A[2][1])*sin(p))*sin(t))
    r4 = lambda t,p: 2*(A[0][0]*cos(p)**2 + (A[0][1] + A[1][0])*cos(p)*sin(p) + A[1][1]*sin(p)**2)*sin(t)**2
    return lambda t,p: (r1(t,p) + r2(t,p))/sqrt(r3(t,p) + r4(t,p))**3

def spheroidFull(A):
    return [ f(A) for f in [spheroid, DspheroidDt, DspheroidDp]]

def oblate(ra,ri,n):
    """ an oblate with outer radius ra and inner radius ri """
    def f(t,p):
        return ra/ri/2*cos(n*2*t) + (ra+ri)/2
    return f

def sprocket(ra,ri,n):
    def f(t,p):
        return ra/ri/2*cos(n*(p-1)) + (ra+ri)/2
    return f

def multiSprocket(ra,ri,n,m):
    def f(t,p):
        return oblate(ra,ri,n)(t,p)*sprocket(ra,ri,m)(t,p)
    return f
