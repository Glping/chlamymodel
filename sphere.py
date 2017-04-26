import surfaceintegration # surfaceGrid, cartesianFromSphere
import tensors            # CompressedTensor
import oseentensors       # polynomTensor, polyTimesOseenDerivative
import flattentensors     # multipole2Vector
import misc               # oddMinus

import numpy              # sin
from numpy.linalg import norm
import math               # factorial

def projectPerpToVector(w, r):
    """ project r onto the plane perpendicular to w """
    r = numpy.array(r)
    w = numpy.array(w)/norm(numpy.array(w))
    return r - numpy.dot(r,w)*w

class Shape(object):
    """
    multipole moments and mobilities vary with their shape.
    this shape dependency, is completely expressed in terms
    of an integration function (surface element and function evaluation).

    This is an abstract class. A subclass only has to provide integration
    functions to offer the full mobility implementation, which is already given
    in this class.
    """
    def __init__(self, *argl, **argd):
        raise NotImplementedError('constructor is not implemented')

    def surfaceArea(self, *argl, **argd):
        raise NotImplementedError('method surfaceArea is not implemented')

    def integrate(self, *argl, **argd):
        raise NotImplementedError('method integrate is not implemented')

    def integrateField(self, *argl, **argd):
        raise NotImplementedError('method integrateField is not implemented')

    def multipoleComponent(self, order, field):
        """
        field is a vector field in cartesian coordinates.
        returns a CompressedTensor.
        """
        res = tensors.CompressedTensor(1,tensors.CompressedTensor(order-1,id=0))
        rHochN = oseentensors.polynomTensor(order-1,self.position)
        coeff = 1/math.factorial(order-1)

        for index in res.indices:
            i = 0 + index[1] + 2*index[2]
            def ifunc(c,ind):
                def hfunc(x,y,z):
                    return field(x,y,z)[ind]*c.evaluate(x,y,z)
                return hfunc

            res.components[index] = tensors.mapTensor(lambda c: coeff*self.integrate(ifunc(c,i)), rHochN)
        return res

    def multipole(self, order, field):
        """
        field is vector field in cartesian coordinates.
        returns a vector, considering the symmetries of the multipole
        (relevant as of third order)
        """
        def getMulti(n):
            hres = self.multipoleComponent(n, field)
            return flattentensors.multipole2Vector(hres)
        return numpy.concatenate([ getMulti(i+1) for i in range(order) ])

    def mobilityComponent(self, n, p, sphere):
        """
        this method returns the mobility between self and sphere.
        Its structure is a highly nested CompressedTensor.
        """
        coeff = misc.oddMinus(p-1)/math.factorial(n-1)
        hhh = oseentensors.polyTimesOseenDerivative(n-1, p-1, self.position, sphere.position)
        res = tensors.mapTensor(lambda c: self.integrate(lambda x,y,z: coeff*c.evaluate(x,y,z)),hhh)
        return res

    def selfmobilityComponent(self, n, p):
        """ CompressedTensor """
        return self.mobilityComponent(n, p, self)

    def mobility(self, order, sphere):
        """
        this method returns the mobility (including selfmobilities)
        up to given order in numpy array structure.
        """
        def oneMob(n,p):
            return flattentensors.mobility2Matrix(self.mobilityComponent(n, p, sphere))
        cols = []
        for n in range(order):
                cols.append( numpy.concatenate([oneMob(n+1, p+1) for p in range(order)], axis=1) )

        res = numpy.concatenate(cols, axis=0)
        return res

    def selfmobility(self, order):
        """ array structure """
        return self.mobility(order, self)



class Sphere(Shape):
    def __init__(self, position=(0,0,0), radius=1, grid=10):
        self.position = position
        self.radius = radius
        oldGrid = surfaceintegration.surfaceGrid(grid)
        self.grid = surfaceintegration.grid2numpySphere(oldGrid, radius)

    def surfaceArea(self):
        return 4*numpy.pi*self.radius*self.radius

    def integrate(self, f):
        """
        surface integration with a homogeneous set of surface points.
        f is a function in cartesian coordinates.
        """
        dAs = self.radius*self.radius*numpy.sin(self.grid[0])*self.grid[2]*self.grid[3]
        fvec = numpy.vectorize(f)
        resH = fvec(self.grid[4]+self.position[0] ,self.grid[5]+self.position[1], self.grid[6]+self.position[2])
        #resH = fvec(self.grid[4] ,self.grid[5], self.grid[6])
        return numpy.sum(dAs*resH)

    def integrateField(self, field):
        """
        surface integration with a homogeneous set of surface points.
        f is a field in cartesian coordinates.
        """
        dAs = self.radius*self.radius*numpy.sin(self.grid[0])*self.grid[2]*self.grid[3]
        fvec = numpy.vectorize(field)
        resH = fvec(self.grid[4]+self.position[0], self.grid[5]+self.position[1], self.grid[6]+self.position[2])*dAs
        #resH = fvec(self.grid[4], self.grid[5], self.grid[6])*dAs
        return (numpy.sum(resH[0]), numpy.sum(resH[1]), numpy.sum(resH[2]))

    def field2velocity(self, f):
        """
        a field is a three dimensional function in cartesian coordinates
        """
        hFunc = lambda x,y,z: f(x+self.position[0], y+ self.position[1], z+self.position[2])
        h = self.integrateField(hFunc)
        coeff = 1/self.surfaceArea()
        return h*coeff


class Ellipsoid(Shape):

    def __init__(self, position=(0,0,0), lengths=(1,1,1), axe1=(1,0,0), axe2=(0,1,0), grid=30):
        """
        an ellipoid at position position with principal axes axe1
        and axe2 with the lengths lengths.
        """
        import shapes
        A = shapes.getSpheroidA(lengths, axe1, axe2)
        (self.r, self.drdt, self.drdp) = map(lambda f: numpy.vectorize(f), shapes.spheroidFull(A))
        self.position = position
        oldGrid = surfaceintegration.surfaceGrid(grid)
        self.grid = surfaceintegration.grid2numpyEllipsoid(oldGrid, self.r)

    def integrate(self, f):
        """
        I want f to be a function of cartesian coordinates!
        """
        prefactor_dA = self.r(self.grid[0],self.grid[1])*numpy.sqrt((self.r(self.grid[0],self.grid[1])**2 + self.drdt(self.grid[0],self.grid[1])**2)*numpy.sin(self.grid[0])**2 + self.drdp(self.grid[0],self.grid[1])**2)
        dAs = prefactor_dA*self.grid[2]*self.grid[3]
        fvec = numpy.vectorize(f)
        res = fvec(self.grid[4] + self.position[0], self.grid[5] + self.position[1], self.grid[6] + self.position[2])*dAs
        return numpy.sum(res)

    def surfaceArea(self):
        return self.integrate(lambda x,y,z: 1)

    def integrateField(self, field):
        """
        a field is a vector function of cartesian coordinates
        """
        prefactor_dA = self.r(self.grid[0],self.grid[1])*numpy.sqrt((self.r(self.grid[0],self.grid[1])**2 + self.drdt(self.grid[0],self.grid[1])**2)*numpy.sin(self.grid[0])**2 + self.drdp(self.grid[0],self.grid[1])**2)
        dAs = prefactor_dA*self.grid[2]*self.grid[3]
        fvec = numpy.vectorize(f)
        res = fvec(self.grid[4] + self.position[0], self.grid[5] + self.position[1], self.grid[6] + self.position[2])*dAs
        return (numpy.sum(res[0]), numpy.sum(res[1]), numpy.sum(res[2]))
