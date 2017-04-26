"""
the next step to the tensor package.
I need an general tensor library where the use of compressed or uncompressed
form is hidden and the methods provide funcionality for numpy array of type
float or object are considered.
Is it even necessary to use numpy arrays?
I try to avoid them, I will provide a function for returning a numpy array.
"""
from misc import permutations,trinomial,oddFac,odd,noverm
from misc import transposeIndex,listQuotient,zipList,zipListWithDict,lessenList
from copy import deepcopy


def inc (alpha):
    """ increment an index """
    if alpha == 0: return 1
    if alpha == 1: return 2
    if alpha == 2: return 0
def dec (alpha):
    """ decrement an index """
    if alpha == 0: return 2
    if alpha == 1: return 0
    if alpha == 2: return 1


def indicesFromCompr (compr):
    """ given an index notation in compressed form return
    a list of indices that correspond to it.
    """
    (n1,n2,n3) = compr
    ind = list( permutations([0]*n1 + [1]*n2 + [2]*n3) )
    return ind
def oneIndexFromCompr (compr):
    (n1,n2,n3) = compr
    return [0]*n1 + [1]*n2 + [2]*n3
def compressedIndices (r):
    """ given a rank, return all possible indexCombinations
    in compressed form. See Applquist 89, chapter 2
    """
    rr = list(range(r+1))
    compr = [ (n3,n2,n1) for n1 in rr for n2 in rr for n3 in rr
                         if n1 + n2 + n3 == r ]
    return compr
def classicIndex2CompressedIndex(index):
    n = {}
    for h in [0,1,2]:
        n[h] = len( [hhh for hhh in index if hhh == h] )
    return (n[0],n[1],n[2])


class CompressedTensor:
    """
    the compressed notation of a symmetric tensor saves very much memory for large
    tensor ranks. Also computations do not repeat.
    Warning: the methods contract and detrace in this class must not be called for
    non numeric tensors containing objects, except the methods __add__ and __rmul__
    are defined.
    the id argument in the contructor is used as an identity object with respect
    to addition.
    """
    def __init__(self,rank,id=None):
        import copy
        self.rank = rank

        rr = list(range(rank+1))
        self.indices = compressedIndices(rank)
        self.components = {}
        self.id = copy.deepcopy(id)
        for i in self.indices:
            self.components[i] = copy.deepcopy(id)

    def __add__(self,tensor):
        if self.rank != tensor.rank:
            print('you try to add two different ranked tensors!')
            print('I give you the greater one...')
            if self.rank > tensor.rank:
                return self
            else:
                return tensor
        res = CompressedTensor(self.rank,id=self.id)
        for i in self.indices:
            res.components[i] = self.components[i] + tensor.components[i]
        return res

    def __rmul__(self,coeff):
        h = CompressedTensor(self.rank,id=self.id)
        for i in self.indices:
            h.components[i] = coeff*self.components[i]
        return h

    def __eq__(self,tensor):
        return compareTensors(self,tensor,0.0001)

    def getComponent(self,index):
        """
        this method is used for duck typing with the ClassicTensor.
        index is an index which would be used by a ClassicTensor.
        The duck typing will be useful in MixedTensors.
        """
        i = classicIndex2CompressedIndex( index )
        return self.components[i]

    def setComponent(self, index, value):
        """
        index is an index in ClassicForm!
        """
        i = classicIndex2CompressedIndex( index )
        self.components[i] = value

    def map(self,func):
        i1 = self.indices[0]
        id = func(self.components[i1])
        h = CompressedTensor(self.rank,id=id)
        for i in self.indices:
            h.components[i] = func(self.components[i])
        return h

    def toClassicForm(self):
        from copy import deepcopy
        res = ClassicTensor(self.rank,id=self.id)
        for key in self.indices:
            for i in indicesFromCompr(key):
                res.setComponent(i,deepcopy(self.components[key]))
        return res

    def flatten(self,id=0):
        """
        call that function if self is a tensor of a tensor.
        get a tensor with rank r1 + r2
        """
        if self.rank == 0:
            return self.components[(0,0,0)]
        testTensor = self.components[(self.rank,0,0)]
        if testTensor.rank == 0:
            return self.map(lambda c: c.components[(0,0,0)])
        hh1 = self.map(lambda c: c.toClassicForm())
        hh2 = hh1.toClassicForm()
        res = hh2.flatten(id)
        return res

    def outer(self,compr):
        """
        outer product of a CompressedTensors with a Compressed or
        a Classic one. Could be a nested CompressedTensor
        to save place, but it is really ugly to work with it.
        the components of both tensors must be combinable
        """
        res = ClassicTensor(self.rank + compr.rank,id=self.id)
        if self.rank == 0:
            return compr.toClassicForm()
        t1 = self.toClassicForm()
        t2 = compr.toClassicForm()
        return t1.outer(t2)

    def contract(self,m):
        """
        the m fold contraction. for a symmetric tensor, the indices
        in which to fold do not matter.
        """
        if m == 0:
            return self
        # rank function for tensor cannot be used
        res = CompressedTensor(self.rank - 2*m,id=self.id)

        # for every index of the contracted tensor
        for (n1,n2,n3) in res.indices:

            res.components[(n1,n2,n3)] = self.id
            for (k1,k2,k3) in compressedIndices(m):

                c = trinomial(k1,k2,k3)
                res.components[(n1,n2,n3)] += c*self.components[(n1+2*k1, n2+2*k2, n3+2*k3)]

        return res

    def detrace(self):
        from math import floor
        res = CompressedTensor(self.rank,id=self.id)
        for (n1,n2,n3) in self.indices:

            hhh = self.id
            m1Max = floor (n1/2)
            m2Max = floor (n2/2)
            m3Max = floor (n3/2)
            for m1 in range(m1Max + 1):
                for m2 in range(m2Max + 1):
                    for m3 in range(m3Max + 1):

                        c = 0
                        if odd(m1+m2+m3): c = -1
                        else:             c = 1
                        c *= oddFac( 2*(n1+n2+n3-m1-m2-m3)-1 )/( oddFac( 2*(n1+n2+n3)-1 ) )
                        contr = self.contract( m1+m2+m3 )
                        h1 = (noverm(n1,m1)*noverm(n2,m2)*noverm(n3,m3) * c)
                        h2 = contr.components[(n1 - 2*m1, n2 - 2*m2, n3 - 2*m3)]
                        hhh += h1 * h2

            res.components[(n1,n2,n3)] = hhh
        return res


def indexCombinations (r):
    if r < 0:
        print ('tensor.indexCombinations: bad usage: rank smaller than one')
        return
    if r == 0:
        return []
    if r == 1:
        return [[0],[1],[2]]
    else:
        return [a+[b] for a in indexCombinations ( r-1 ) for b in [0,1,2]]
def symIndexCombinations (r):
    """given a rank r of a tensor, computes all possible combinations of
    indices needed by a symmetric tensor"""
    if r == 1:
        return [[0],[1],[2]]
    else:
        return [[b]+a for a in symIndexCombinations(r-1) for b in [0,1,2] if b <= a[0]]



class ClassicTensor:

    def __init__(self,rank,id=None):
        self.rank = rank
        self.id = id
        self.indices = indexCombinations(rank)
        self.components = self.fillComponents(rank,id)

    def fillComponents(self,r,id):
        from copy import deepcopy
        if r == 1:
            return [deepcopy(id),deepcopy(id),deepcopy(id)]
        return [self.fillComponents(r-1,id) for i in range(3)]

    def getComponent(self,index):
        h = self.components
        for i in index:
            h = h[i]
        return h

    def setComponent(self,index,value):
        h = self.components
        for i in index[:-1]:
            h = h[i]
        h[index[-1]] = value

    def map(self,func):
        i1 = self.indices[0]
        id = func(self.getComponent(i1))
        res = ClassicTensor(self.rank,id=id)
        def mapH(ll,r):
            if r == 0:
                return func(ll)
            else:
                return [ mapH(l,r-1) for l in ll ]
        res.components = mapH(self.components,self.rank)
        return res

    def toCompressedForm(self):
        res = CompressedTensor(self.rank)
        for i in res.indices:
            res.components[i] = self.getComponent(oneIndexFromCompr(i))
        return res

    def toClassicForm(self):
        return self

    def show(self,showfunction=None):
        """
        nice representation of a Tensor.
        only applicable for number!
        """
        def showH(comps,rank):
            if rank == 1:
                (x,y,z) = comps
                if showfunction != None:
                    print('| {0} | {1} | {2} |'.format(showfunction(x)
                                                      ,showfunction(y)
                                                      ,showfunction(z)))
                else:
                    print('| {0:9f} | {1:9f} | {2:9f} |'.format(x,y,z))
            elif rank == 2:
                for i in range(3):
                    showH(comps[i],1)
            else:
                for i in range(3):
                    showH(comps[i],rank-1)
                    print('-------------------------------------')
        showH(self.components,self.rank)

    def __add__(self,tensor):
        if self.rank != tensor.rank:
            print('you try to add tensors of different rank!')
            raise TensorRankError('adding different ranked tensors!')
            if self.rank > tensor.rank:
                return self
            else:
                return tensor
        res = ClassicTensor(self.rank,id=self.id)
        for i in self.indices:
            res.setComponent(i,self.getComponent(i) + tensor.getComponent(i))
        return res

    def __eq__(self,tensor):
        return compareTensors(self,tensor,0.0001)

    def __rmul__(self,coeff):
        h = ClassicTensor(self.rank,id=self.id)
        for i in self.indices:
            h.setComponent(i, coeff*self.getComponent(i))
        return h

    def transpose(self,i1,i2):
        res = ClassicTensor(self.rank,id=self.id)
        for i in res.indices:
            res.setComponent( i, self.getComponent(transposeIndex(i,i1,i2)))
        return res

    def subtensor(self,ii):
        if self.rank - len(ii) <= 0:
            res = self.getComponent([ii[0]])
        else:
            res = ClassicTensor(self.rank - len(ii),id=self.id)
            for i in res.indices:
                res.setComponent( i, self.getComponent( zipListWithDict(i, ii) ) )
        return res

    def flatten(self,id=0):
        """ use this, if self is ClassicTensor of Tensors """
        subtensor = self.getComponent(self.indices[0])
        res = ClassicTensor(self.rank+subtensor.rank, id=id)
        hhh = self.map(lambda t: t.toClassicForm())
        for i in res.indices:
            res.setComponent(i, hhh.getComponent(i[:self.rank]).getComponent(i[self.rank:]) )
        return res



    def outer(self,tensor):
        """ outer product of two tensor resulting in a tensor of summed rank """
        res = ClassicTensor(self.rank+tensor.rank,id=self.id)
        t2 = tensor.toClassicForm()
        for (i,j) in [ (a,b) for a in self.indices for b in t2.indices ]:
            res.setComponent( i+j, self.getComponent(i) * t2.getComponent(j) )
        return res

    def contract (self,index1,index2):
        if self.rank == 0:
            return self
        elif self.rank == 2:
            res = self.getComponent([0,0])+self.getComponent([1,1])+self.getComponent([2,2])
        else:
            res = ClassicTensor(self.rank-2, id=self.id)
            for i in range(3):
                h = self.subtensor({index1:i,index2:i})
                res += h
        return res

    def symmetricIndices(self):
        return symIndexCombination(self.r)

    def symmetrize (self):
        """
        due to the unnecessity to return a ClassicTensor, a
        CompressedTensor is returned here.
        """
        res = CompressedTensor(self.rank,id=self.id)
        for i in res.indices:
            sum = self.id
            perms = indicesFromCompr(i)
            np = len (perms)
            for index in perms:
                sum += self.getComponent(index)
            res.components[i] = (1/np)*sum
        return res

    def tracelessSymmetric (self):
        return self.symmetrize().detrace()

    def cycleIndex(self,n=1):
        """ put the last n indices to the front """
        res = ClassicTensor(self.rank,id=self.id)
        for i in self.indices:
            ni = i[-n:] + i[:-n]
            res.setComponent(ni,self.getComponent(i))
        return res

    def shiftIndexLeft(self,width,num=1):
        """ shift the rightmost index num positions to the left """
        res = ClassicTensor(self.rank,id=self.id)
        for i in self.indices:
            hi = i
            for n in range(num):
                i2 = [hi[-1]]
                i3 = hi[-(width+1):-1]
                i1 = hi[:self.rank-width-1]
                hi = i1+i2+i3
            res.setComponent(hi,self.getComponent(i))
        return res

    def numpy(self,dtype=float):
        from numpy import array
        return array(self.components,dtype=dtype)

class MixedTensor:
    """
    two kinds of tensors are known so far: ClassicTensor and CompressedTensor.
    Both classes are able to produce instances of the other one.
    Now imagine a tensor symmetrized in only a few indizes. It would be convenient
    to have mixed tensor of rank rcl+rcp, that means a ClassicTensor of rank rcl with
    components CompressedTensors of rank rcp.
    Another example of a mixed tensor would be a symmetrized tensor of symmetrized
    tensor components. The problem we have to deal with is the order of the indices.
    We need to keep track of how indizes changed for beeing able to reproduce a Classical
    tensor, which is the purest description that will be used for converting our tensors
    to numpy arrays.
    The constructor returns a ClassicTensor of ClassicTensors.
    symmetrizeInner returns a ClassicTensor of CompressedTensors.
    symmetrizeOuter returns a CompressedTensor of ClassicTensors.
    tracelessSymmetricInner returns a ClassicTensor of CompressedTensors.
    tracelessSymmetricOuter returns a CompressedTensor of ClassicTensors.
    """
    def __init__(self,tensor,indexPositions):
        """
        divide a ClassicTensor into a smaller tensor of tensors given by indexpositions.
        keep track of the backtransformation.
        indexPositions are transposed to the end
        """
        self.id = tensor.id
        self.outer = ClassicTensor( tensor.rank - len(indexPositions)
                                  , id=ClassicTensor(len(indexPositions),id=tensor.id))
        self.indexPositions = indexPositions
        for outerIndex in self.outer.indices:
            h = ClassicTensor(len(indexPositions), id=tensor.id)
            for innerIndex in h.indices:
                h.setComponent( innerIndex
                              , tensor.getComponent( zipList(outerIndex,innerIndex,indexPositions) ) )
            self.outer.setComponent( outerIndex, h )

    def symmetrizeInner(self):
        if self.outer.getComponent( self.outer.indices[0] ).rank <= 1:
            return self
        h = ClassicTensor( self.outer.rank
                         , id=CompressedTensor( len(self.indexPositions)
                                              , id=self.id ) )
        for i in h.indices:
            h.setComponent( i, self.outer.getComponent( i ).symmetrize() )
        self.outer = h
        return self

    def symmetrizeOuter(self):
        if self.outer.rank <= 1:
            return self
        h = CompressedTensor( self.outer.rank
                            , id=self.outer.id )
        h = self.outer.symmetrize()
        self.outer = h
        return self

    def tracelessSymmetricInner (self):
        if self.outer.getComponent( self.outer.indices[0] ).rank <= 1:
            return self
        h = ClassicTensor( self.outer.rank
                         , id=CompressedTensor( len(self.indexPositions)
                                              , id=self.id ) )
        for i in h.indices:
            h.setComponent( i, self.outer.getComponent( i ).tracelessSymmetric() )
        self.outer = h
        return self

    def tracelessSymmetricOuter (self):
        if self.outer.rank <= 1:
            return self
        h = CompressedTensor( self.outer.rank
                            , id=self.outer.id )
        h = self.outer.tracelessSymmetric()
        self.outer = h
        return self

    def toClassicForm(self):
        res = ClassicTensor(self.outer.rank + len(self.indexPositions), id=self.id)
        houter = self.outer.toClassicForm()
        for io in houter.indices:

            h = houter.getComponent(io).toClassicForm()
            for ii in h.indices:

                res.setComponent( zipList(io,ii,self.indexPositions), houter.getComponent(io).getComponent(ii) )
        return res


######################################################################
#
# some factory functions

def fromList(ll):
    def h(ll,n):
        if type(ll) != list and type(ll) != tuple:
            return (ll,n)
        else:
            return h(ll[0],n+1)
    (val,rank) = h(ll,0)
    res = ClassicTensor(rank, id=val)
    res.components = list(ll)
    return res

def fromNumpy(tensor,id=0):
    rank = len(tensor.shape)
    res = ClassicTensor(rank,id=id)
    for i in res.indices:
        res.setComponent(i, tensor[tuple(i)])
    return res

def identity():
    """ identity matrix """
    res = ClassicTensor(2,id=0)
    for i in [0,1,2]:
        res.setComponent([i,i],1)
    return res

def leviCivita():
    """ the 3 dimensional well known synbol"""
    res = ClassicTensor(3,id=0)
    for i in [(0,1,2),(1,2,0),(2,0,1)]:
        res.setComponent(i,1)
    for i in [(2,1,0),(1,0,2),(0,2,1)]:
        res.setComponent(i,-1)
    return res

def randomTensor(r):
    from random import random
    t = ClassicTensor(r,id=0)
    for i in t.indices:
        t.setComponent(i,random())
    return t

######################################################################
######################################################################
#
# usable
#

def contract2(tensor1, is1, tensor2, is2, id=0):
    """
    is1 and is2 are index positions.
    if tensor1 and tensor2 are both rank 1 tensors, return a scalar.
    """
    resRank = tensor1.rank + tensor2.rank - 2
    if tensor1.rank == 1 and tensor2.rank == 1:
        res = id
        for i in [0, 1, 2]:
            res += tensor1.getComponent([i]) * tensor2.getComponent([i])
        return res
    res = ClassicTensor(resRank, id=id)
    for i in [0,1,2]:
        sub1 = tensor1.subtensor({is1:i})
        sub2 = tensor2.subtensor({is2:i})
        if tensor1.rank == 1:
            res += mapTensor(lambda c: sub1 * c, sub2)
        elif tensor2.rank == 1:
            res += mapTensor(lambda c: sub2 * c, sub1)
        else:
            res += sub1.outer(sub2)
    return res


def mapTensor(func,tensor):
    """
    this function maps over a tensor. It is recursive and decide if it is
    looking at a component by testing for the rank property.
    -> works only if components does not have a rank property!
    """
    if hasattr(tensor,'rank'): # tensor is a tensor
        return tensor.map(lambda c: mapTensor(func,c))
    else: # tensor is a component
        return func(tensor)

def flattenTensor(tensor,id=0):
    """ flatten an arbitraryly nested tensor """
    if tensor.rank == 0:
        return flattenTensor(tensor.components[(0,0,0)],id)
    anIndex = tensor.indices[0]
    if type(anIndex) == list: # if tensor is a ClassicTensor
        subtensor = deepcopy(tensor.getComponent(anIndex))
    elif type(anIndex) == tuple: # if tensor is a CompressedTensor
        subtensor = deepcopy(tensor.components[anIndex])
    else:
        print('bad usage of flattenTensor!')
        return
    if hasattr(subtensor,'rank'):
        return flattenTensor(tensor.flatten(id),id)
    else:
        return tensor

def reduceTensor(f,id,tensor):
    """
    folding a tensor
    example: f(x,y): x+y:
    sum up all components of a tensor.
    """
    flat = flattenTensor(tensor)
    res = id

    for i in flat.indices:
        if type(i) == tuple:
            res = f(res,flat.components[i])
        else:
            res = f(res,flat.getComponent(i))
    return res

def evaluateTensor(tensor,x,y,z):
    """
    evaluate all components of a tensor. This is possible for
    arbitrarily nested tensors, because Compressed AND Classic
    tensors have the map method.
    the components must have a evaluate method, that takes three
    arguments.
    A tensor of exact same shape is returned
    """
    # if it is a component, evaluate
    if hasattr(tensor,'evaluate'):
        return tensor.evaluate(x,y,z)
    # if it is a tensor, map over it
    else:
        h = tensor.map(lambda t: evaluateTensor(t,x,y,z))
        h.id = 0
        return h


def compareTensors(t1,t2,eps=0.001):
    if t1.rank != t2.rank:
        return False
    for i in t1.indices:
        d = t1.getComponent(i) - t2.getComponent(i)
        if d < 0 and d < (-1)*eps:
            return False
        if d > 0 and d > eps:
            return False
    return True
