from numpy import array, sin, cos, pi, sqrt
from numpy import zeros, ones, concatenate, cumsum, dot
from numpy.linalg import solve
from scipy.integrate import cumtrapz


################################################################################
#
# resistive force theory. Its essence is encapsulated inside `forceDensity`.
# data collection routine is getState
#
def forceDensity(tx,ty,nx,ny,vx,vy, zPar=0.7, zPer=1.25):
    fx = zPar*(vx*tx + vy*ty)*tx + zPer*(vx*nx + vy*ny)*nx
    fy = zPar*(vx*tx + vy*ty)*ty + zPer*(vx*nx + vy*ny)*ny
    return (fx, fy)


def frictionMatrixHead(dx, dy, chlamy=False):
    """
    friction matrix for the head of a swimmer, which is assumed to be
    an ellipse. Therefore the analytical solution of Perron can be used.
    It is further set, that the swimmer is described in the matrial frame.
    """
    from perrin import translateThin, translateBroad, rotateThin
    K = zeros((3,3))
    if chlamy:
        K[0,0] = translateThin(dx, dy)
        K[1,1] = translateBroad(dx, dy)
    else:
        K[0,0] = translateBroad(dx, dy)
        K[1,1] = translateThin(dx, dy)
    K[2,2] = rotateThin(dx, dy)
    return K


################################################################################
#
# PDO <-> plain old data structure. nicer usage than that one of dictionaries
# but same purpose
#
# now, extended, added some analysing methods
#
class Flagelle(object):
    """
    description of the flagellum in the material frame
    """

    def __init__(self, psi, dpsi, tx, ty, nx, ny, xflag, yflag, vx, vy, fx, fy, V, F):
        self.psi = psi
        self.dpsi = dpsi
        self.nx = nx
        self.ny = ny
        self.tx = tx
        self.ty = ty
        self.xflag = xflag
        self.yflag = yflag
        self.vx = vx
        self.vy = vy
        self.fx = fx
        self.fy = fy
        self.V = V
        self.F = F
    
    ################################################################################
    #
    # analysing stuff
    #
    def velocityDistributionInMaterialFrame(self):
        """
        returned values assume that the flagellum is at the origin of the
        material frame.
        flagellum: instance of the class Flagellum, defined above
        """
        vxM = self.vx + self.V[0] - self.yflag*V[2]
        vyM = self.vy + self.V[1] + self.xflag*V[2]
        return (vxM, vyM)
    
    def velocitiesInMaterialFrame(self):
        (vxM, vyM) = self.velocityDistributionInMaterialFrame()
        return (sum(vxM), sum(vyM), sum(vyM*self.xflag - vxM*self.yflag))

    def centerOfMass(self):
        """ center of mass in the material frame """
        num = len(self.xflag)
        return (sum(self.xflag)/num, sum(self.yflag)/num)

#
################################################################################
class Flagella(object):
    """
    purpose: capture flagella data for a whole trajectory
    also provide some methods for analysing quantities in
    lab and material frame.
    """
    def __init__(self, flags, dt):
        self.num = len(flags)
        self.spatialResolution = len(flags[0].xflag)
        #print(self.spatialResolution)
        self.dt = dt
        self.psi = array([flags[i].psi for i in range(self.num)])
        self.xflag = array([flags[i].xflag for i in range(self.num) ])
        self.yflag = array([flags[i].yflag for i in range(self.num) ])
        self.vx = array([flags[i].vx for i in range(self.num) ])
        self.vy = array([flags[i].vy for i in range(self.num) ])
        self.fx = array([flags[i].fx for i in range(self.num) ])
        self.fy = array([flags[i].fy for i in range(self.num) ])
        self.V = [flags[i].V for i in range(self.num) ]
        self.Vx = array([v[0] for v in self.V])
        self.Vy = array([v[1] for v in self.V])
        self.Wz = array([v[2] for v in self.V])
        self.F = [flags[i].F for i in range(self.num) ]
        self.Fx = array([f[0] for f in self.F])
        self.Fy = array([f[1] for f in self.F])
        self.Tz = array([f[2] for f in self.F])

    def centerOfMass(self):
        num = len(self.xflag)
        return ([sum(x)/num for x in self.xflags],
                [sum(y)/num for y in self.yflags])

    def append(self, f):
        self.num += f.num
        self.psi = concatenate([self.psi, f.psi])
        self.xflag = concatenate([self.xflag, f.xflag])
        self.yflag = concatenate([self.yflag, f.yflag])
        self.vx = concatenate([self.vx, f.vx])
        self.vy = concatenate([self.vy, f.vy])
        self.fx = concatenate([self.fx, f.fx])
        self.fy = concatenate([self.fy, f.fy])
        self.V = concatenate([self.V, f.V])
        self.Vx = concatenate([self.Vx, f.Vx])
        self.Vy = concatenate([self.Vy, f.Vy])
        self.Wz = concatenate([self.Wz, f.Wz])
        self.F = concatenate([self.F, f.F])
        self.Fx = concatenate([self.Fx, f.Fx])
        self.Fy = concatenate([self.Fy, f.Fy])
        self.Tz = concatenate([self.Tz, f.Tz])




################################################################################
#
# the force balancing procedure

def integrateForceDensity(xflag, yflag, fx, fy, ds):
    Fx = sum(fx)*ds
    Fy = sum(fy)*ds
    Mz = (dot(fy,xflag) - dot(fx,yflag))*ds
    return array([Fx, Fy, Mz])


def psi2tangent(psi):
    return (cos(psi), sin(psi))

def psi2normal(psi):
    (tx, ty) = psi2tangent(psi)
    return (-ty, tx)

def tangent2normal(tx, ty):
    return (-ty, tx)

def tangent2path(tx, ty, ds):
    xflag = concatenate([array([0]), cumtrapz(tx)])*ds
    yflag = concatenate([array([0]), cumtrapz(ty)])*ds
    return (xflag, yflag)

def psi2path(psi, ds):
    (tx, ty) = psi2tangent(psi)
    (xflag, yflag) = tangent2path(tx, ty, ds)
    return (xflag, yflag)

def getState(psi, dpsi, ds, head=None):
    spatialResolution = len(psi)
    #print(psi.shape)
    # tangent and normal have length 1 \mu m each
    (tx, ty) = psi2tangent(psi)
    (nx, ny) = tangent2normal(tx, ty)
    (xflag, yflag) = tangent2path(tx, ty, ds)

    K_flag = zeros((3,3))

    vx = ones((spatialResolution))  # in \mu m / ms
    vy = zeros((spatialResolution))
    (fx, fy) = forceDensity(tx,ty,nx,ny,vx,vy)
    K_flag[:,0] = integrateForceDensity(xflag,yflag,fx,fy, ds)

    vx = zeros((spatialResolution))
    vy = ones((spatialResolution))
    (fx, fy) = forceDensity(tx,ty,nx,ny,vx,vy)
    K_flag[:,1] = integrateForceDensity(xflag,yflag,fx,fy, ds)

    vx = -yflag
    vy = xflag
    (fx, fy) = forceDensity(tx,ty,nx,ny,vx,vy)
    K_flag[:,2] = integrateForceDensity(xflag,yflag,fx,fy, ds)

    if head is not None:
        (dx, dy) = head
        K_flag = K_flag + frictionMatrixHead(dx, dy)

    vx = concatenate([array([0]), cumtrapz(nx*dpsi)])*ds
    vy = concatenate([array([0]), cumtrapz(ny*dpsi)])*ds
    (fx, fy) = forceDensity(tx,ty,nx,ny,vx,vy)
    F = integrateForceDensity(xflag,yflag,fx,fy, ds)

    V = -solve(K_flag, F)

    return Flagelle(psi, dpsi, tx, ty, nx, ny, xflag, yflag, vx, vy, fx, fy, V, F)

def getStateMixedDerivative(ps1, ps2, dps, ds, head=None):
    """
    for stochastic processes, the choice of derivative is very important.
    two version are calculated and the average of both are taken.
    """
    flagellum1 = getState(ps1, dps, ds, head=head)
    flagellum2 = getState(ps2, dps, ds, head=head)
    ps = (ps1 + ps2) / 2
    (tx, ty) = psi2tangent(ps2)
    (nx, ny) = tangent2normal(tx, ty)
    (xflag, yflag) = tangent2path(tx, ty, ds)
    vx = (flagellum1.vx + flagellum2.vx) / 2
    vy = (flagellum1.vy + flagellum2.vy) / 2
    fx = (flagellum1.fx + flagellum2.fx) / 2
    fy = (flagellum1.fy + flagellum2.fy) / 2
    V = (flagellum1.V + flagellum2.V) / 2
    F = (flagellum1.F + flagellum2.F) / 2
    return Flagelle(ps, dps, tx, ty, nx, ny, xflag, yflag,
                    vx, vy, fx, fy, V, F)


#
################################################################################

################################################################################
#
# tests:
#

#
## a standing wave does not cause propulsion
#

if __name__ == '__main__':
    import resistiveForce
    from plotting.plot2d import linlin
    from plotting.animate import trajectoryWithArrowsAnim
    # physical parameters
    length = 60      # length of the flagellum in \mu m
    omega = 3*pi/50  # angular velocity in ms^{-1}
    period = 2*pi/omega            # current time in ms
    numberOfPeriods = 1
    (zPar, zPer) = (0.5,0.5)
    # simulation parameters
    timeResolution = 500
    spatialResolution = 100
    # derived parametes
    ds = length/spatialResolution
    dt = period/timeResolution
    times = [ dt*t for t in range(timeResolution*numberOfPeriods)]

    #### isotropic friction coefficients on a flagellum without head, there is no propulsion
    #
    print('travelling wave, isotropic friction')

    flagella = [ symmetricSwimmer(t, omega, length, spatialResolution) for t in times ]

    Vx =  array([ f.V[0] for f in flagella ])
    Vy =  array([ f.V[1] for f in flagella ])
    Vz =  array([ f.V[2] for f in flagella ])
    Xx =  array([ f.xflag for f in flagella ])
    Yy =  array([ f.yflag for f in flagella ])
    Fxx = array([ f.fx for f in flagella ])
    Fyy = array([ f.fy for f in flagella ])
    trajectoryWithArrowsAnim(Xx, Yy, Fxx, Fyy)
    (rx, ry) = reproduceTrajectory(Vx, Vy, Vz, dt)
    linlin([rx],[ry], title='trajectory of the head')

    #### a standing wave does not cause propulsion
    #
    zPer = 1
    print('standing wave, nonisotropic friction')
    flagella = [ standingWaveSwimmer(t, omega, length, spatialResolution) for t in times ]

    Vx =  array([ f.V[0] for f in flagella ])
    Vy =  array([ f.V[1] for f in flagella ])
    Vz =  array([ f.V[2] for f in flagella ])
    Xx =  array([ f.xflag for f in flagella ])
    Yy =  array([ f.yflag for f in flagella ])
    Fxx = array([ f.fx for f in flagella ])
    Fyy = array([ f.fy for f in flagella ])
    trajectoryWithArrowsAnim(Xx, Yy, Fxx, Fyy)
    (rx, ry) = reproduceTrajectory(Vx, Vy, Vz, dt)
    linlin([rx],[ry], title='trajectory')
    linlin([times]*3,[Vx, Vy, Vz], xlabel='time in $ms$', ylabel='velocity in $\mu m ms^{-1}$ or $ms^{-1}$')

    #### length of the flagellum is proportional to propulsion
    #
    print('symmetric flagellum: varying length')
    d = []
    ls = [20,30,40,50,60]
    for l in ls:
        length = l
        ds = length/spatialResolution
        flagella = [ symmetricSwimmer(t, omega, length, spatialResolution) for t in times ]
        Vx =  array([ f.V[0] for f in flagella ])
        Vy =  array([ f.V[1] for f in flagella ])
        Vz =  array([ f.V[2] for f in flagella ])
        (rx, ry) = reproduceTrajectory(Vx, Vy, Vz, dt)
        d.append(sqrt((rx[-1] - rx[0])**2 - (ry[-1] - ry[0])**2))

    linlin([ls], [d], xlabel='length of flagellum in $\mu m$', ylabel='distance after one period in $\mu m$')

    #### rotation after period is proportional to k*period
    #
    print('bended swimmer: rotation after period is proportional to k')
    a = []
    ks = [0.001, 0.002, 0.003, 0.004, 0.005]
    for k in ks:
        flagella = [ asymmetricSwimmer(t,k, omega, length, spatialResolution) for t in times ]
        Vx =  array([ f.V[0] for f in flagella ])
        Vy =  array([ f.V[1] for f in flagella ])
        Vz =  array([ f.V[2] for f in flagella ])
        Xx =  array([ f.xflag for f in flagella ])
        Yy =  array([ f.yflag for f in flagella ])
        Fxx = array([ f.fx for f in flagella ])
        Fyy = array([ f.fy for f in flagella ])
        alphas = cumsum(Vz*dt)
        a.append(alphas[-1] - alphas[0])
        (rx, ry) = reproduceTrajectory(Vx, Vy, Vz, dt)
        linlin([rx],[ry], title='trajectory')
    linlin([ks], [a], xlabel='bending value', ylabel='angle difference')
    trajectoryWithArrowsAnim(Xx, Yy, Fxx, Fyy)
