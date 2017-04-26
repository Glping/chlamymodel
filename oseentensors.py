"""
the oseen tensor and its derivatives are provided
and several other tensors containing Polynoms or
OseenTerms or OseenFunctions such as multipoles and
shape moments.
"""
from math import pi

from tensors import ClassicTensor,CompressedTensor,MixedTensor
from tensors import mapTensor, flattenTensor
from oseenfunctions import OseenFunction,Polynom,OseenTerm


def oseenTensor(eta=0.7):
    """
    due to the symmetry of the oseen tensor a tensor in compressed
    form should be returned?
    The derivatives of the oseen tensor are also symmetric to each other
    but the indizes of the oseen tensor and the derivatives are not.
    """
    res = CompressedTensor(2,id=Polynom(0,0,0,0))
    c = 1/8/pi/eta
    for (nx,ny,nz) in res.indices:

        p = Polynom(c,nx,ny,nz)
        t = OseenTerm(3,p)
        #res[i,j] = OseenFunction([t])
        if 2 in (nx,ny,nz):
            q = Polynom(c,0,0,0)
            s = OseenTerm(1,q)
            res.components[(nx,ny,nz)] = OseenFunction([s,t])
        else:
            res.components[(nx,ny,nz)] = OseenFunction([t])

    return res


def poseenTensor(eta=0.7):
    c = 1 / (4 * pi)
    res = CompressedTensor(1, id=Polynom(0, 0, 0, 0))
    res.setComponent([0], OseenFunction([OseenTerm(3, Polynom(c, 1, 0, 0))]))
    res.setComponent([1], OseenFunction([OseenTerm(3, Polynom(c, 0, 1, 0))]))
    res.setComponent([2], OseenFunction([OseenTerm(3, Polynom(c, 0, 0, 1))]))
    return res


def nablaOseen():
    o = oseenTensor()
    res = CompressedTensor( 2, id=OseenFunction([]))
    for i in res.indices:
        h = res.components[i]
        for d in [0,1,2]:
            h += o.components[i].nDerive([d,d])
        res.components[i] = h
    return res


def compressedOseenFunctionTensorDerivative(t, n):
    """
    the nth derivative of a compressed tensor, containing elements
    that implement a nDerive method.
    The result is a compressed tensor of compressed tensors.
    """
    if n == 0:
        return t
    res = CompressedTensor(n,id=CompressedTensor(t.rank,id=OseenFunction([])))#OseenTerm(0,Polynom(0,0,0,0))])))
    for (dx,dy,dz) in res.indices:
        h = CompressedTensor(t.rank,id=OseenFunction([]))#OseenTerm(0,Polynom(0,0,0,0))]))
        for i in t.indices:
            h.components[i] = t.components[i].nDerive([0]*dx+[1]*dy+[2]*dz)
        res.components[(dx,dy,dz)] = h
    return res


def nablaOseenDerivative(n):
    """
    nth derivative of the nabla of the Oseen tensor
    """
    no = nablaOseen()
    return compressedOseenFunctionTensorDerivative(no,n)


def oseenDerivative(n):
    """
    the nth derivative of the oseen tensor.
    it is a CompressedTensor of a CompressedTensor.
    """
    o = oseenTensor()
    return compressedOseenFunctionTensorDerivative(o,n)


def poseenDerivative(n):
    """
    the nth derivative of the poseen tensor.
    it is a CompressedTensor of a CompressedTensor.
    """
    o = poseenTensor()
    return compressedOseenFunctionTensorDerivative(o,n)


def symmetrizedGradient(v):
    """
    v: rank 1 ClassicTensor of Oseenfunctions
    returns:
        \Gamma_{ij} = 1 / 2 * (\partial_i v_j + \partial_j v_i)
        rank 2 CompressedTensor of Oseenfunctions.
    first create the velocitygradient, then symmetrize.
    """
    h = compressedOseenFunctionTensorDerivative(v.toCompressedForm(), 1)
    return flattenTensor(h, OseenFunction([])).symmetrize()




def evaluateOseenDerivative(deriv,x,y,z):
    """
    oseenDerivatives are CompressedTensors of CompressedTensors.
    Thus, we map map them
    """
    h = deriv.map( lambda t: t.evaluate(x,y,z))
    return h


def polynomTensor(p,r0=(0,0,0)):
    """
    a tensor that contains polynoms (r-r0)^p.
    Such a tensor is symmetric, therefore the polynoms are saved within a
    compressed tensor.
    """
    res = CompressedTensor(p,id=Polynom(0,0,0,0))
    if r0 == (0,0,0):
        for (nx,ny,nz) in res.indices:
            res.components[(nx,ny,nz)] = Polynom(1,nx,ny,nz)
    else:
        for (nx,ny,nz) in res.indices:
             x = (Polynom(1,1,0,0)-Polynom(r0[0],0,0,0))**nx
             y = (Polynom(1,0,1,0)-Polynom(r0[1],0,0,0))**ny
             z = (Polynom(1,0,0,1)-Polynom(r0[2],0,0,0))**nz
             res.components[(nx,ny,nz)] = x*y*z

    return res


def polyTimesOseenDerivative(n,p,r0=(0,0,0),r1=(0,0,0)):
    """
    this results in a tensor of functions which is used very often: multipoles and mobility
    the outer product of a tensor of polynoms with the oseenDerivatives.
    This is a very efficient way to compute it because symmetries of tensors are used.
    """
    rhn = polynomTensor(n,r0)
    ode = mapTensor(lambda c: c.shift(r1),oseenDerivative(p))
    # as a first step create a CompressedTensor of rank rhn.rank
    # (because it is much smaller than ode.flatten().rank) with id
    # ode.flatten().id (OseenFunctions)
    # hhh will be a CompressedTensor of a CompressedTensor of a CompressedTensor.
    # As a result we need to perform much fewer integrations
    # TODO: this kind of optimization must be made in explicitMultipoles!
    res = CompressedTensor( rhn.rank
                          , id=CompressedTensor( p
                                               , id=CompressedTensor( 2
                                                                    , OseenFunction({}).shift(r1))))
    for r in rhn.indices:
        oFunc = OseenFunction([OseenTerm(0,rhn.components[r])])
        res.components[r] = mapTensor(lambda c: c*oFunc,ode)

    return res
