"""
if we know psi and dpsi/dt (and some parameters), we are able to compute velocities
and forces. Some are only for testing purposes, others for real data examination.
"""

from math import pi
from numpy import array, sin, cos

###############################################################################
#
# function that provide the flagellar path in terms of the angle description
# psi for a certain time.

def reverseSwimmer(psi, dpsi):
    """
    given the discretization of the tangent angles of a flagellum,
    reversing it and adding pi.
    The derivative has to be inversed only.
    """
    psiRev = psi[::-1] + pi
    dpsiRev = dpsi[::-1]
    return (psiRev, dpsiRev)

def straight(omega):
    psi = lambda s, t: sin(omega*t)
    dpsi = lambda s, t: omega*cos(omega*t)
    return (psi, dpsi)

def artificialPsiAndDpsi(omega, lmbd, A=1):
    """
    time (t) dependent values of psi at parametrization point (s).
    omega: phase velocity
    lmbd: length scale of the beat
    A: Amplitude. for A << 1: x(T) - x(0) \prop A^2
    return a travelling wave and its time derivative.
    """
    lmbdH = 2*pi/float(lmbd)
    def func(s,t):
        return A*(cos(s*lmbdH - omega*t) - cos(omega*t))
    def dfunc(s,t):
        return A*omega*(sin(s*lmbdH - omega*t) + sin(omega*t))
    return (func, dfunc)

def scallop(omega, length, bending):
    def func(s, t):
        return bending * (s - length / 2) * cos(omega * t)
    def dfunc(s, t):
        return -omega * bending * (s - length / 2) * sin(omega * t)
    return (func, dfunc)

def bendedWave(omega, lmbd, k=0.01, A=1):
    (f1, f2) = artificialPsiAndDpsi(omega, lmbd, A=A)
    return (lambda s,t: f1(s,t) + k*s, f2)

def bensBendedWave(omega, lmbd, k=0.01, A=1):
    lmbdH = 2*pi/lmbd
    psi = lambda s, t: k*s + 2*A*s*cos(omega*t - lmbdH*s)
    dpsi = lambda s, t: -2*s*A*omega*sin(omega*t - lmbdH*s)
    return (psi, dpsi)

def standingWave(omega, lmbd, A=1):
    """ standing wave and its time derivative """
    lmbdH = 2*pi/float(lmbd)
    def func(s,t):
        return A*cos(omega*t)*cos(lmbdH*s)
    def dfunc(s,t):
        return -omega*A*sin(omega*t)*cos(lmbdH*s)
    return (func, dfunc)

def reconstructFlagellum(psi0, psi1, psi2, psi3, omega, t):
    """
    bens data filtering produces psi_i values
    can be reconstructed here.
    """
    phi = omega*t
    psi  = psi0 + 1j*0
    psi += (psi1['re'] + 1j*psi1['im'])*exp(1j*phi)
    psi += (psi1['re'] - 1j*psi1['im'])*exp(-1j*phi)
    psi += (psi2['re'] + 1j*psi2['im'])*exp(2j*phi)
    psi += (psi2['re'] - 1j*psi2['im'])*exp(-2j*phi)
    psi += (psi3['re'] + 1j*psi3['im'])*exp(3j*phi)
    psi += (psi3['re'] - 1j*psi3['im'])*exp(-3j*phi)
    dpsi  = (psi1['re'] + 1j*psi1['im'])*exp(1j*phi)*omega*1j
    dpsi += (psi1['re'] - 1j*psi1['im'])*exp(-1j*phi)*omega*(-1j)
    dpsi += (psi2['re'] + 1j*psi2['im'])*exp(2j*phi)*2j*omega
    dpsi += (psi2['re'] - 1j*psi2['im'])*exp(-2j*phi)*(-2j)*omega
    dpsi += (psi3['re'] + 1j*psi3['im'])*exp(3j*phi)*3j*omega
    dpsi += (psi3['re'] - 1j*psi3['im'])*exp(-3j*phi)*(-3j)*omega
    return (real(psi), real(dpsi))

###############################################################################
#
# function that provide the flagellar path in terms of the angle description
# psi for a certain time PERIOD.

def stochasticPhase(omega, length, spatialResolution, timeResolution, dt, D=2.8, A=0.1):
    """
    simulate a stochastic process, where the phase contains random.
    Therefore the flagellum has to be prepared for the whole period,
    not only the for one time step.
    """
    from diffusion import DiffusiveParticle
    lmbd = 2 * pi / length
    ds = length / spatialResolution

    dp = DiffusiveParticle(D, lambda x: omega, x0=0, dt=dt)
    phi = [h[0] for h in dp.steps(timeResolution + 1)]

    def psifunc(s, phi):
        return A * (cos(lmbd * s - phi) - cos(phi))

    psis = []
    for p in phi:
        h = array([psifunc(i * ds, p) for i in range(spatialResolution)])
        psis.append(h)
    dpsis = []
    for i in range(len(psis) - 1):
        dpsis.append((psis[i + 1] - psis[i]) / dt)
    return (psis[:-1], dpsis)

def stochasticAmplitude(omega, length, spatialResolution, timeResolution, dt,
                        A=5, sigmaA=1, tau=1):
    """
    simulate a stochastic flagellar path where the amplitude of the
    beat is random.
    """
    from diffusion import DiffusiveParticle
    lmbd = 2 * pi / length
    dp = DiffusiveParticle(sigmaA * sigmaA / tau, lambda x: (1 - x) / tau, x0=1, dt=dt)
    ramp = [h[0] for h in dp.steps(timeResolution + 1)]

    def psifunc(s, t):
        return A * (cos(lmbd * s - omega * t) - cos(omega * t))
    def dpsifunc(s, t):
        return A * omega * (sin(lmbd * s - omega * t) + sin(omega * t))

    psis = []
    dpsis = []
    for (a, i) in zip(ramp, range(timeResolution + 1)):
        ds = length / spatialResolution
        t = dt * i
        psis.append(array([a * psifunc(ds * s, t) for s in range(spatialResolution)]))
        # dpsis.append([a * dpsifunc(ds * s, t) for s in range(spatialResolution)])

    for i in range(timeResolution):
        dpsis.append((psis[i + 1] - psis[i]) / dt)

    return (psis, dpsis)
