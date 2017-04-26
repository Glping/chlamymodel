"""
applications of the force resistive theory for tests and data analysis.
"""
from RFT.procedure import *
from RFT.shapes import *

from math import pi


def functionalSwimmer(t, psiFunc, dpsiFunc, ds, spatialResolution, head=None):
    psi = array([ psiFunc(s*ds, t) for s in range(spatialResolution + 1)])
    dpsi = array([ dpsiFunc(s*ds, t) for s in range(spatialResolution + 1)])
    return getState(psi, dpsi, ds, head=head)


def reversedFunctionalSwimmer(t, psiFunc, dpsiFunc, ds, spatialResolution, head=None):
    psi = array([ psiFunc(s*ds, t) for s in range(spatialResolution + 1)])
    dpsi = array([ dpsiFunc(s*ds, t) for s in range(spatialResolution + 1)])
    (psi, dpsi) = reverseSwimmer(psi, dpsi)
    return getState(psi, dpsi, ds, head=head)


def straightSwimmer(t, omega, length, spatialResolution):
    (psi, dpsi) = straight(omega)
    ds = length/spatialResolution
    return functionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=None)


def dicretizedDerivativeSwimmer(t, psiFunc, dpsiFunc, ds, spatialResolution, head=None):
    psi = array([ psiFunc(s*ds, t) for s in range(spatialResolution)])
    psi2 = array([ psiFunc(s*ds, t+0.00001) for s in range(spatialResolution)])
    dpsi = (psi2 - psi)/0.00001
    return getState(psi, dpsi, ds, head=head)


def scallopSwimmer(t, bending, omega, length, spatialResolution, head=None):
    (psi, dpsi) = scallop(omega, length, bending)
    ds = length / spatialResolution
    return functionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=head)


def symmetricSwimmer(t, omega, length, spatialResolution, head=None):
    ds = length/spatialResolution
    (psi, dpsi) = artificialPsiAndDpsi(omega, length, A=0.5)
    return functionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=head)


def symmetricSwimmerData(t, omega, length, spatialResolution):
    ds = length/spatialResolution
    (psiFunc, dpsiFunc) = artificialPsiAndDpsi(omega, length, A=0.5)
    psi = array([ psiFunc(s * ds, t) for s in range(spatialResolution + 1)])
    (xflag, yflag) = psi2path(psi, ds)
    return {'xflag': xflag, 'yflag': yflag, 'head': (5, 2.5)}


def reversedSymmetricSwimmer(t, omega, length, spatialResolution, head=None):
    ds = length/spatialResolution
    (psi, dpsi) = artificialPsiAndDpsi(omega, length, A=0.5)
    return reversedFunctionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=head)


def asymmetricSwimmer(t, k, omega, length, spatialResolution, head=None):
    """ asymmetry through bending with curvature k """
    ds = length/spatialResolution
    (psi, dpsi) = bendedWave(omega, length, k=k, A=0.5)
    return functionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=head)


def reversedAsymmetricSwimmer(t, k, omega, length, spatialResolution, head=None):
    """ asymmetry through bending with curvature k """
    ds = length/spatialResolution
    (psi, dpsi) = bendedWave(omega, length, k=k, A=0.5)
    return reversedFunctionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=head)


def bensBendedSwimmer(t, k, omega, length, A, head, spatialResolution):
    """ asymmetry through bending with curvature k """
    ds = length/spatialResolution
    (psi, dpsi) = bensBendedWave(omega, length, k=k, A=A)
    return functionalSwimmer(t, psi, dpsi, ds, spatialResolution, head=head)


def standingWaveSwimmer(t, omega, length, spatialResolution):
    ds = length/spatialResolution
    (psi, dpsi) = standingWave(omega, length, A=0.5)
    return functionalSwimmer(t, psi, dpsi, ds, spatialResolution)


def bensSwimmer(t, omega, length, spatialResolution, folder='.', head=(10,5)):
    from os import chdir, getcwd
    currentFolder = getcwd()
    chdir(folder)
    psi0 = loadtxt('psi0.dat')
    psi1 = { 're': loadtxt('psi1re.dat'), 'im': loadtxt('psi1im.dat') }
    psi2 = { 're': loadtxt('psi2re.dat'), 'im': loadtxt('psi2im.dat') }
    psi3 = { 're': loadtxt('psi3re.dat'), 'im': loadtxt('psi3im.dat') }
    chdir(currentFolder)
    (psi, dpsi) = reconstructFlagellum(psi0, psi1, psi2, psi3, omega, t)
    ds = length/len(psi)
    return getState(psi, dpsi, ds, head=head)


# the following to swimmer are identical to the symmetric swimmer,
# except for fluctuations in their phase and amplitude, respectively.


def stochasticPhaseSwimmer1(omega, length, spatialResolution, timeResolution, dt,
                           D=2.8, A=0.1, head=None):
    (psi, dpsi) = stochasticPhase(omega, length, spatialResolution, timeResolution, dt, D=D, A=A)
    ds = length / spatialResolution
    flagella = []
    for (ps, dps) in zip(psi, dpsi):
        flagella.append(getState(ps, dps, ds, head=head))
    return Flagella(flagella, dt)

def stochasticPhaseSwimmer2(omega, length, spatialResolution, timeResolution, dt,
                            D=2.8, A=0.1, head=None):
    (psi, dpsi) = stochasticPhase(omega, length, spatialResolution, timeResolution + 1, dt, D=D, A=A)
    ds = length / spatialResolution
    flagella = []
    for (ps, dps) in zip(psi[1:], dpsi[:-1]):
        flagella.append(getState(ps, dps, ds, head=head))
    return Flagella(flagella, dt)

def stochasticPhaseSwimmer(omega, length, spatialResolution, timeResolution, dt,
                           D=2.8, A=0.1, head=None):
    (psi, dpsi) = stochasticPhase(omega, length, spatialResolution, timeResolution + 1, dt, D=D, A=A)
    ds = length / spatialResolution

    flagella1 = []
    for (ps, dps) in zip(psi, dpsi):
        flagella1.append(getState(ps, dps, ds, head=head))
    flagella2 = []
    for (ps, dps) in zip(psi[1:], dpsi[:-1]):
        flagella2.append(getState(ps, dps, ds, head=head))

    return (Flagella(flagella1, dt), Flagella(flagella2, dt))


def stochasticAmplitudeSwimmer1(omega, length, spatialResolution, timeResolution, dt,
                               A=5, sigmaA=1, tau=1, head=None):
    (psi, dpsi) = stochasticAmplitude(omega, length, spatialResolution, timeResolution, dt,
                                      A=A, sigmaA=sigmaA, tau=tau)
    ds = length / spatialResolution
    flagella = []
    for (ps, dps) in zip(psi, dpsi):
        flagella.append(getState(ps, dps, ds, head=head))
    return Flagella(flagella, dt)

def stochasticAmplitudeSwimmer2(omega, length, spatialResolution, timeResolution, dt,
                                A=5, sigmaA=1, tau=1, head=None):
    (psi, dpsi) = stochasticAmplitude(omega, length, spatialResolution, timeResolution + 1, dt,
                                      A=A, sigmaA=sigmaA, tau=tau)
    ds = length / spatialResolution
    flagella = []
    for (ps, dps) in zip(psi[1:], dpsi[:-1]):
        flagella.append(getState(ps, dps, ds, head=head))
    return Flagella(flagella, dt)

def stochasticAmplitudeSwimmer(omega, length, spatialResolution, timeResolution, dt,
                               A=5, sigmaA=0.035, tau=5.9, head=None):
    (psi, dpsi) = stochasticAmplitude(omega, length, spatialResolution, timeResolution, dt,
                                      A=A, sigmaA=sigmaA, tau=tau)
    ds = length / spatialResolution
    flagella1 = []
    for (ps, dps) in zip(psi, dpsi):
        flagella1.append(getState(ps, dps, ds, head=head))
    flagella2 = []
    for (ps, dps) in zip(psi[1:], dpsi[:-1]):
        flagella2.append(getState(ps, dps, ds, head=head))

    return (Flagella(flagella1, dt), Flagella(flagella2, dt))


def eulerHeunSwimmer(psi, dpsi, ds, dt, head=None):
    flagella1 = []
    for (ps, dps) in zip(psi, dpsi):
        flagella1.append(getState(ps, dps, ds, head=head))
    flagella2 = []
    for (ps, dps) in zip(psi[1:], dpsi[:-1]):
        flagella2.append(getState(ps, dps, ds, head=head))

    return (Flagella(flagella1, dt), Flagella(flagella2, dt))


def eulerHeunSwimmerWithoutDpsi(psi, ds, dt, head=None):
    flagella1 = []
    flagella2 = []
    for (ps1, ps2) in zip(psi, psi[1:]):
        dpsi = (ps2 - ps1) / dt
        flagella1.append(getState(ps1, dpsi, ds, head=head))
        flagella2.append(getState(ps2, dpsi, ds, head=head))

    return (Flagella(flagella1, dt), Flagella(flagella2, dt))
    


# omega = 2 * pi / 50
# period = 2 * pi / omega
# numberOfPeriods = 1
# time = period * numberOfPeriods
# timeResolution = 1000 * numberOfPeriods
# spatialResolution = 100
# dt = time / timeResolution
# length = 60
# ds = length / spatialResolution
# 
# D = 0.000028
# A = 0.005
# sigmaA = 0.01
# tau = dt * 10
# 0.00002
# 
# # flagella = stochasticPhaseSwimmer(omega,
# #                                   length,
# #                                   spatialResolution,
# #                                   timeResolution,
# #                                   dt,
# #                                   A=A,
# #                                   D=D)
# 
# flagella = stochasticAmplitudeSwimmer(omega,
#                            length,
#                            spatialResolution,
#                            timeResolution,
#                            dt,
#                            A=A,
#                            sigmaA=sigmaA,
#                            tau=tau,
#                            head=None)
# import plotting.animate as ani
# 
# (xflag, yflag) = flagella.flagellaPathLabFrame()
# (vx, vy, fx, fy) = flagella.distributionsInLabFrame()
# ani.trajectoryWithArrowsAnim(xflag,
#                              yflag,
#                              vx,
#                              vy,
#                              oneOf=5)
# 
# (cx, cy, alpha) = flagella.positionOfMaterialFrame()
# import plotting.plot2d as plt
# plt.linlin([cx], [cy])
