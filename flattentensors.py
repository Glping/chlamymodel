"""
tensors are nice objects for hiding the internal complexity of computations
and structure the mathematics in a nice way.
The nice formalism builded up from these abbreviations, however, are not useful
for using performant numerical libraries. For performing the task of solving a
linear system of equation, a flat matrix and e vector is needed.
This module provides functions for converting tensors from tensors.py into
numpy arrarys or matrices, which can then be used for the task just mentioned.

Due to the purpose of this module, all tensors are assumed to be
arbitrarily nested tensors of NUMBERS.
"""

from tensors import flattenTensor
from tensors import ClassicTensor
from tensors import CompressedTensor
from tensors import randomTensor
from tensors import compressedIndices,indexCombinations
from tensors import indicesFromCompr

from misc import repeat

from numpy import zeros

def symmetricNumber(i):
    """
    given an index in compressed notation, return the number of
    realizations in conventional tensor notation.
    """
    if i == (0,0,0):
        return 1
    return len(indicesFromCompr(i))

def symmetricVSflatIndices(n):
    """
    given a rank n of a tensor return a dictionary
    that maps vector indices to indices of a CompressedTensor
    """
    res = {}
    indices = compressedIndices(n)
    num = len(indices)
    for (i,c) in zip(range(num),indices):
        res[i] = c
    return res

def multipole2Vector(tensor):
    """
    higher order moments of multipoles have an intrinsic symmetry.
    Due to this, they are saved as CompressedTensors of CompressedTensors.
    Thus, the argument tensor is such an object.
    A naive reduction to vector form leads to much more components.
    The reduction scheme in this function takes symmetries into account.
    """
    f = tensor.components # f is a dictionary with CompressedTensors of rank (moment - 1)
    indices = symmetricVSflatIndices(f[(1,0,0)].rank) # indices are the same for all components
    l = len(indices)
    res = zeros([3*l])
    for fi in [(0,(1,0,0)),(1,(0,1,0)),(2,(0,0,1))]:
        for i in indices:
            res[i + fi[0]*l] = f[fi[1]].components[indices[i]]
    return res

def vector2Multipole(vector,n):
    """ undo multipole2Vector result in a n multipole """
    f = CompressedTensor(1,id=CompressedTensor(n-1,id=0))
    indices = symmetricVSflatIndices(n-1)
    l = len(indices)
    for fi in [(0,(1,0,0)),(1,(0,1,0)),(2,(0,0,1))]:
        for i in indices:
            f.components[fi[1]].components[indices[i]] = vector[i + fi[0]*l]
    return f

def vector2ForceMultipole(vector,n):
    return vector2Multipole(vector,n)

def vector2VelocityMultipole(vector,n):
    """
    because of a symmetry shift, due to the application of the g-matrix
    velocity and force multipoles cannot be treated in the same way!
    """
    f = CompressedTensor(1,id=CompressedTensor(n-1,id=0))
    indices = symmetricVSflatIndices(n-1)
    l = len(indices)
    for fi in [ (0,(1,0,0)), (1,(0,1,0)), (2,(0,0,1)) ]:
        for (i,ind) in zip(range(len(indices)),indices):
            f.components[fi[1]].components[indices[i]] = vector[3*i + fi[0]]
    return f

def mobility2Matrix(tensor):
    """
    higher order mobility matrices should be flattened here. Two symmetries
    are taken into account during construction: commutativity of multiplication
    and commutativity of differentiation. Mobility tensors are CompressedTensors
    of CompressedTensors of CompressedTensors.
    """
    n = tensor.rank
    nIndices = symmetricVSflatIndices(n)
    p = tensor.components[tensor.indices[0]].rank
    pIndices = symmetricVSflatIndices(p)
    pl = len(pIndices)
    nl = len(nIndices)
    res = zeros([3*nl,3*pl])
    for ni in nIndices:
        for pi in pIndices:
            for i in [0,1,2]:
                for j in [0,1,2]:
                    rowIndex = ni + i*nl
                    colIndex = pi + j*pl
                    res[(rowIndex,colIndex)] = tensor.components[nIndices[ni]
                                                    ].components[pIndices[pi]
                                                    ].getComponent([i,j])*symmetricNumber(pIndices[pi])
    return res

def flatIndex(i):
    """
    given an index of the ClassicTensor notation
    return a flat index for numpy arrays (one-dimensional)
    """
    sm = 0
    for (elem,l) in zip(i,range(len(i))):
        sm += 3**l*elem
    return sm

def rankFromNumberOfElements(N):
    def hhh(n,r):
        if n < 1:
            print('rankFromNumberOfElements called with a bad argument')
        elif n == 1:
            return r
        else:
            return hhh(n/3,r+1)
    return hhh(N,0)

def vector2Tensor(vector,revers=True):
    """
    the inverse of tensor2Vector.
    take the non symmetrized vector variant.
    """
    rank = rankFromNumberOfElements(len(vector))
    res = ClassicTensor(rank,id=0)
    for i in res.indices:
        if revers:
            res.setComponent(list(reversed(i)), vector[flatIndex(i)])
        else:
            res.setComponent(i, vector[flatIndex(i)])
    return res

def tensor2Vector(tensor,revers=True):
    """
    velocity and force multipoles have to be converted.
    produces the non symmetrized variant of vectors (larger)
    """
    tensor = flattenTensor(tensor).toClassicForm()
    numberOfElements = 3**tensor.rank
    res = zeros([numberOfElements])
    for i in tensor.indices:
        if revers:
            res[flatIndex(i)] = tensor.getComponent(list(reversed(i)))
        else:
            res[flatIndex(i)] = tensor.getComponent(i)
    return res

def velocity2Vector(tensor):
    return tensor2Vector(tensor,revers=True)

def force2Vector(tensor,revers=False):
    return tensor2Vector(tensor,revers=False)

def tensor2Matrix(tensor,down): # same as toMatrix, redundant
    """
    convert a tensor to a matrix. down gives the number of indices
    that are used to produce rows.
    """
    right = tensor.rank - down
    rowNum = 3**down
    colNum = 3**right
    res = zeros([rowNum,colNum])
    downInds = indexCombinations(down)
    rightInds = indexCombinations(right)
    for (ci,i) in zip(range(len(downInds)),downInds):
        for (cj,j) in zip(range(len(rightInds)),rightInds):

            res[(ci,cj)] = tensor.getComponent( list(reversed(i))+j )

    #for i in tensor.indices:
    #    rowi = flatIndex(i[:down])
    #    coli = flatIndex(i[down:])
    #    res[(rowi,coli)] = tensor.getComponent(i)
    return res

def toMatrix(tensor,down):
    """
    the first 'down' indices produce rows,
    the others produce columns
    works only for ClassicTensors
    """
    res = zeros([3**down,3**(tensor.rank-down)])
    for i in tensor.indices:
        counter = 0
        indexX = 0
        indexY = 0
        for ip in reversed(i):
            if 3**counter < 3**down:
                indexX += ip*(3**counter)
            else:
                indexY += ip*(3**(counter-down))
            counter += 1
        res[(indexX,indexY)] = tensor.getComponent(reversed(i))
    return res

def fromMatrix(matrix,rank):
    """
    given a matrix (of shape x or (x,y)), produce a ClassicTensor
    of rank rank.
    """
    def threetothepowerofdown(n):
        if n == 1:
            return 0
        elif n == 3:
            return 1
        else:
            return 1 + threetothepowerofdown(n/3)
    res = ClassicTensor(rank,id=0)
    down = threetothepowerofdown(matrix.shape[0])
    for i in res.indices:
        counter = 0
        indexX = 0
        indexY = 0
        for ip in reversed(i):
            if 3**counter < 3**down:
                indexX += ip*(3**counter)
            else:
                indexY += ip*(3**(counter-down))
            counter += 1
        if len(matrix.shape) == 1:
            res.setComponent(list(reversed(i)),matrix[max(indexX,indexY)])
        else:
            res.setComponent(list(reversed(i)),matrix[(indexX,indexY)])
    return res

def printMatrix(m):
    shape = m.shape
    ii = shape[0]
    if len(shape) == 2:
        try:
            jj = shape[1]
        except:
            jj = 1
        for i in range(ii):
            for j in range(jj):
                print('{0: 02.3f} '.format(m[i,j]),end='')
            print('')
    else:
        for i in range(ii):
            print('{0: 02.3f}'.format(m[i]))


def indexVector(m):
    """
    this vector shows, which indices are built into the bigVector
    """
    from numpy import array
    indices = symmetricVSflatIndices(m-1) # indices are the same for all components
    l = len(indices)
    res = array(['']*(3*l),dtype=object)
    for fi in [(0,(1,0,0)),(1,(0,1,0)),(2,(0,0,1))]:
        for i in indices:
            res[i + fi[0]*l] = str([indices[i]])+str(fi[1])
    return res

def indexMatrix(n,p):
    """
    same as indexVector, but for a matrix
    """
    from numpy import array
    nIndices = symmetricVSflatIndices(n)
    pIndices = symmetricVSflatIndices(p)
    pl = len(nIndices)
    nl = len(pIndices)
    res = array([['']*(3*pl)]*(3*nl),dtype=object)
    print(res.shape)
    for ni in nIndices:
        for pi in pIndices:
            for i in [0,1,2]:
                for j in [0,1,2]:
                    rowIndex = ni + i*nl
                    colIndex = pi + j*pl
                    res[(rowIndex,colIndex)] = str(nIndices[ni])+str(pIndices[pi])+str([i,j])
    return res
