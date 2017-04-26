"""
polynoms are sums of with coefficients multiplied products of variables.
here I implement them using an alternative data structure, which can
be used to derive analytical derivatives.
"""
from copy import deepcopy

# [(3,(3,1,2),(1,(0,0,4))] is equivalent to 3*x*x*x*y*z*z + z*z*z*z
# better: a dict: {(3,1,2):3,(0,0,4):1}
class Polynom:

    def __init__(self,c,nx,ny,nz):
        self.intern = {}
        if c != 0:
            self.intern[(nx,ny,nz)] = c

    def addTerm(self,c,nx,ny,nz):
        if (nx,ny,nz) in self.intern:
            if self.intern[(nx,ny,nz)] == -c:
                self.intern.pop((nx,ny,nz))
            else:
                self.intern[(nx,ny,nz)] += c
        else:
            self.intern[(nx,ny,nz)] = c

    def isNull(self):
        res = True
        for n in self.intern:
            res &= self.intern[n] == 0
        return res

    def negativ(self):
        p = Polynom(-1,0,0,0)
        return self.mult(p)

    def mult(self,polynom):
        h = Polynom(0,0,0,0)
        for (nx,ny,nz) in self.intern:
            for (px,py,pz) in polynom.intern:
                c = self.intern[(nx,ny,nz)]*polynom.intern[(px,py,pz)]
                h.addTerm(c,nx+px,ny+py,nz+pz)
        return h
    def add(self,polynom):
        h = Polynom(0,0,0,0)
        for (nx,ny,nz) in self.intern:
            if self.intern[(nx,ny,nz)] != 0:
                h.addTerm(self.intern[(nx,ny,nz)],nx,ny,nz)
        for (kx,ky,kz) in polynom.intern:
            if polynom.intern[(kx,ky,kz)] != 0:
                h.addTerm(polynom.intern[(kx,ky,kz)],kx,ky,kz)
        return h
    def __add__(self,polynom):
        return self.add(polynom)

    def __mul__(self,pol):
        res = Polynom(0,0,0,0)
        for (sx,sy,sz) in self.intern:
            for (px,py,pz) in pol.intern:
                res.addTerm( self.intern[(sx,sy,sz)]*pol.intern[(px,py,pz)],sx+px,sy+py,sz+pz )
        return res

    def __sub__(self,polynom):
        return self + polynom.negativ()

    def __pow__(self,n):
        if n == 0:
            return Polynom(1,0,0,0)
        res = self
        for i in range(n-1):
            res *= self
        return res

    def __rmul__(self,coeff):
        """
        warning: this multiplication is multiplication with a number
        not with another polynom
        """
        p = Polynom(0,0,0,0)
        for (nx,ny,nz) in self.intern:
            p.addTerm(coeff*self.intern[(nx,ny,nz)],nx,ny,nz)
        return p

    def shift(self,origin):
        """
        transform a polynom: p(r) -> p(r-r0)
        return tansformed polynom, keep the old one
        """
        res = Polynom(0,0,0,0)
        for (nx,ny,nz) in self.intern:
            n = self.intern[(nx,ny,nz)]
            x = (Polynom(1,1,0,0)-Polynom(origin[0],0,0,0))**nx
            y = (Polynom(1,0,1,0)-Polynom(origin[1],0,0,0))**ny
            z = (Polynom(1,0,0,1)-Polynom(origin[2],0,0,0))**nz
            res += n*x*y*z
        return res

    def evaluate(self,x,y,z):
        val = 0
        for (nx,ny,nz) in self.intern:
            h = self.intern[(nx,ny,nz)]
            if nx > 0:
                h *= x**nx
            if ny > 0:
                h *= y**ny
            if nz > 0:
                h *= z**nz
            val += h
        return val

    def deriveTerm(self,direction,nx,ny,nz):
        c = self.intern[(nx,ny,nz)]
        if direction == 0:
            if nx > 0:
                term = (nx-1,ny,nz)
                c *= nx
            else:
                return False
        elif direction == 1:
            if ny > 0:
                term = (nx,ny-1,nz)
                c *= ny
            else:
                return False
        else: # direction == 2
            if nz > 0:
                term = (nx,ny,nz-1)
                c *= nz
            else:
                return False
        return (term,c)

    def derive(self,direction):
        """ direction is 0 for x, 1 for y, 2 for z """
        pol = Polynom(0,0,0,0)
        for (nx,ny,nz) in self.intern:
            h = self.deriveTerm(direction,nx,ny,nz)
            if h == False:
                continue
            ((x,y,z),coeff) = h
            pol.addTerm(coeff,x,y,z)
        return pol

    def nDerive(self,directions):
        h = self
        for d in directions:
            h = h.derive(d)
        return h

    def show(self):
        h = ''
        for (nx,ny,nz) in self.intern:
            h += str(self.intern[(nx,ny,nz)])+'*'
            for x in range(nx):
                h += 'x*'
            for y in range(ny):
                h += 'y*'
            for z in range(nz):
                h += 'z*'
            h = h[:-1]+' + '
        return h[:-3]

    def latex(self):
        h = ''
        for (nx,ny,nz) in self.intern:
            h += '{0:.2f}'.format(self.intern[(nx,ny,nz)])
            for x in range(nx):
                h += 'x'
            for y in range(ny):
                h += 'y'
            for z in range(nz):
                h += 'z'
            h = h+' + '
        return h[:-3]

class OseenTerm:
    """ a common term in the oseen tensor is a polynom multiplied by 1/r^n """
    def __init__(self,n,polynom):
        self.n = n
        self.polynom = polynom
        self.origin = (0,0,0)

    def derive(self,direction):
        if direction == 0:
            pol = Polynom(-1*self.n,1,0,0)
        if direction == 1:
            pol = Polynom(-1*self.n,0,1,0)
        if direction == 2:
            pol = Polynom(-1*self.n,0,0,1)
        h1 = OseenTerm(self.n+2,self.polynom.mult(pol))
        h2 = OseenTerm(self.n,self.polynom.derive(direction))
        return [h1,h2]

    def evaluate(self,x,y,z):
        from math import sqrt
        (u,v,w) = self.origin
        r = sqrt((x-u)*(x-u) + (y-v)*(y-v) + (z-w)*(z-w))
        if r == 0:
            print ('warning: divided by zero, return a large value ...')
            return 100000000000
        return self.polynom.evaluate(x,y,z)/(r**self.n)

    def show(self):
        h = '( '+self.polynom.show()+' )'
        if self.origin == (0,0,0):
            h += '/(r**'+str(self.n)+')'
        else:
            h += '/((r-'+str(self.origin)+')**'+str(self.n)+')'
        return h

    def latex(self):
        h = '\\frac{'+self.polynom.latex()+'}'
        if self.origin == (0,0,0):
            h += '{r^{'+str(self.n)+'}}'
        else:
            ho = str(self.origin[0])+'\\\\\n'+str(self.origin[1])+'\\\\\n'+str(self.origin[2])+'\n'
            h += '{\\left|\mathbf{r}-\\left(\\begin{smallmatrix}'+ho+'\\end{smallmatrix}\\right)\\right|^{'+str(self.n)+'}}'
        return h

class OseenFunction:
    """ to initialze a OseenFunction we use a list of OseenTerms """
    def __init__(self,ots):
        self.intern = {}
        for ot in ots:
            #if ot.n == 0:
            #        continue
            if ot.n in self.intern:
                self.intern[ot.n] = self.intern[ot.n].add(ot.polynom)
            else:
                self.intern[(ot.n)] = ot.polynom
        # shift is only used fo the mobility calculation
        # be careful at evaluation!
        self.origin = (0,0,0)

    def __add__(self,of):
        h = OseenFunction([])
        for n in of.intern:
            if not of.intern[n].isNull():
                h.intern[n] = deepcopy(of.intern[n])
        for n in self.intern:
            if n in h.intern:
                h.intern[n] += self.intern[n]
            else:
                if not self.intern[n].isNull():
                    h.intern[n] = deepcopy(self.intern[n])
        h.origin = self.origin
        return h

    def __mul__(self,of):
        res = []
        for n1 in self.intern:
            for n2 in of.intern:
                res.append( OseenTerm(n1+n2,self.intern[n1]*of.intern[n2]) )
        rres = deepcopy(OseenFunction(res))
        rres.origin = self.origin
        return rres

    def __rmul__(self,coeff):
        h = OseenFunction([])
        for n in self.intern:
            h.intern[n] = coeff*self.intern[n]
        return h.shift(self.origin)

    def shift(self,origin):
        """
        shift the function by a position: x/r -> (x-x0)/r'
        the polynom data structure is fitted, but not r'.
        r' must be taken care of at evaluation time. Set a
        property origin for that purpose.
        """
        res = OseenFunction([])
        h0 = self.origin[0] + origin[0]
        h1 = self.origin[1] + origin[1]
        h2 = self.origin[2] + origin[2]
        res.origin = (h0,h1,h2)
        for n in self.intern:
            res.intern[n] = self.intern[n].shift((h0,h1,h2))
        res.origin = (h0,h1,h2)
        return res

    def derive(self,direction):
        hs = []
        for n in self.intern:
            h = OseenTerm(n,self.intern[n])
            (d1,d2) = h.derive(direction)
            if len(d1.polynom.intern) > 0:
                hs.append(d1)
            if len(d2.polynom.intern) > 0:
                hs.append(d2)
        return OseenFunction(hs)

    def nDerive(self,directions):
        h = self
        for d in directions:
            h = h.derive(d)
        return h

    def evaluate(self,x,y,z):
        h = 0
        for k in self.intern:
            ho = OseenTerm(k,self.intern[k])
            ho.origin = self.origin
            h += ho.evaluate(x,y,z)
        return h

    def show(self):
        h = ''
        for n in self.intern:
            ho = OseenTerm(n,self.intern[n])
            ho.origin = self.origin
            h += ho.show()+' + '
        return h[:-3]

    def latex(self):
        h = ''
        for n in self.intern:
            ho = OseenTerm(n,self.intern[n])
            ho.origin = self.origin
            h += ho.latex()+' + '
        return h[:-3]
