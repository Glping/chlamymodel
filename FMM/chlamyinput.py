from flagella.reconstruct import readData, data2rftInit, data2rftPsis
import RFT.procedure as procedure
import numpy as np
from scipy.interpolate import interp1d
import sys
import misc


def osci_fit(x_x, y_y):
    def template(x, a, b_1, b_2, c_1, c_2, d_1, d_2,
                 e_1, e_2, f_1, f_2, g_1, g_2, h_1, h_2):
        res = a + b_1 * np.sin(x) + b_2 * np.cos(x)
        res += c_1 * np.sin(2 * x) + c_2 * np.cos(2 * x)
        res += d_1 * np.sin(3 * x) + d_2 * np.cos(3 * x)
        res += e_1 * np.sin(4 * x) + e_2 * np.cos(4 * x)
        res += f_1 * np.sin(5 * x) + f_2 * np.cos(5 * x)
        res += g_1 * np.sin(6 * x) + g_2 * np.cos(6 * x)
        res += h_1 * np.sin(7 * x) + h_2 * np.cos(7 * x)
        return res
    params = misc.curve_fit(template, x_x, y_y)[0]
    return lambda x: template(x, *params)



def ruloff_chlamy_curvature(which, phase, curvature, spatial_resolution=20):

    oldpath = sys.path
    sys.path.append('/home/gary/MPI/data/ruloff/chosen')
    data_module = __import__(which)
    sys.path = oldpath

    ### make modes functions of arc length
    arcs = np.linspace(0, data_module.length, len(data_module.mode_0))
    mode_0 = interp1d(arcs, data_module.mode_0)
    mode_1 = interp1d(arcs, data_module.mode_1)
    mode_2 = interp1d(arcs, data_module.mode_2)

    ### make scores periodic functions
    phis = np.linspace(0, 2 * np.pi, 100)
    beta_1 = process.osci_fit(phis, data_module.scores_1)
    beta_2 = process.osci_fit(phis, data_module.scores_2)

    ### tangent angle representation
    arcs = np.linspace(0, data_module.length, spatial_resolution)
    psi = mode_1(arcs) * beta_1(phase)
    psi += mode_2(arcs) * beta_2(phase)
    psi += curvature * mode_0(arcs)

    ### cartesian coordinates
    ds = data_module.length / spatial_resolution
    xflag_l = -data_module.dbx - np.cumsum(np.sin(psi)) * ds
    yflag_l = data_module.dby + np.cumsum(np.cos(psi)) * ds
    xflag_r = data_module.dbx + np.cumsum(np.sin(psi)) * ds
    yflag_r = data_module.dby + np.cumsum(np.cos(psi)) * ds

    return {'body': (data_module.minor, data_module.major),
            'xflagR': xflag_r,
            'yflagR': yflag_r,
            'xflagL': xflag_l,
            'yflagL': yflag_l,
            'psiL': psi,
            'psiR': psi,
            'length': data_module.length}


def ruloff_chlamy(
        which, phase, amplitude, spatial_resolution=20, shortening=0.9):
    """
    which: filename with chlamydomonas data.
    phase and amplitude are the two parameters, describing the flagellum.
    spatial_resolution is the number of points to be returned.
    shortening: the data in the files actually miss the tips
    (1 - shortening) * 100 percent of the flagella.
    """

    oldpath = sys.path
    sys.path.append('/home/gary/MPI/data/ruloff/chosen')
    data_module = __import__(which)
    sys.path = oldpath

    ### make modes functions of arc length
    arcs = np.linspace(0, data_module.length, len(data_module.mode_0))
    mode_0 = interp1d(arcs, data_module.mode_0)
    mode_1 = interp1d(arcs, data_module.mode_1)
    mode_2 = interp1d(arcs, data_module.mode_2)

    ### make scores periodic functions
    phis = np.linspace(0, 2 * np.pi, 100)
    beta_1 = process.osci_fit(phis, data_module.scores_1)
    beta_2 = process.osci_fit(phis, data_module.scores_2)

    ### tangent angle representation
    init_spatial_resolution = int(spatial_resolution * shortening)
    arcs = np.linspace(0, data_module.length, init_spatial_resolution)
    psi = mode_1(arcs) * beta_1(phase)
    psi += mode_2(arcs) * beta_2(phase)
    psi *= amplitude
    psi += mode_0(arcs)
    rest_spatial_resolution = spatial_resolution - init_spatial_resolution
    psi_fit = misc.linear_fit([1, 2], psi[-2:])[2]
    psi = np.concatenate(
        [psi, psi_fit([i + 3 for i in range(rest_spatial_resolution)])])

    def psi_func(s, p, a):
        psi = mode_1(s) * beta_1(p)
        psi += mode_2(s) * beta_2(p)
        psi *= a
        return psi + mode_0(s)

    ### cartesian coordinates
    ds = 7 / 30 * data_module.length / spatial_resolution
    xflag_l = 7 / 30 * -data_module.dbx - np.cumsum(np.sin(psi)) * ds
    yflag_l = 7 / 30 * data_module.dby + np.cumsum(np.cos(psi)) * ds
    xflag_r = 7 / 30 * data_module.dbx + np.cumsum(np.sin(psi)) * ds
    yflag_r = 7 / 30 * data_module.dby + np.cumsum(np.cos(psi)) * ds

    return {'body': (7 / 30 * data_module.minor, 7 / 30 * data_module.major),
            'dbx': (7 / 30 * data_module.dbx),
            'dby': (7 / 30 * data_module.dby),
            'xflagR': xflag_r,
            'yflagR': yflag_r,
            'xflagL': xflag_l,
            'yflagL': yflag_l,
            'psiL': psi,
            'psiR': psi,
            'psi_func': psi_func,
            'modes': [m(arcs) for m in [mode_0, mode_1, mode_2]],
            'length': 7 / 30 * data_module.length}


def flagellaAndBody(timeResolution=100,
                    spatialResolution=60,
                    omega=50,
                    cut=0,
                    amplitude=1,
                    timesteps=[]):

    dt = 2 * np.pi / (omega * timeResolution)
    times = [i * dt for i in timesteps]

    data = readData('/home/gary/MPI/bin/resistiveForce/bensMatlab/data')
    dInit = data2rftInit(data)

    # body is an ellipse
    body = dInit.body

    # where are the falgella docked?
    bx = dInit.bx
    by = dInit.by

    # tangent angles and normals in material frame
    hhh = [data2rftPsis(data, omega, time, amplitude=amplitude)
               for time in times]
    psiR = np.zeros((len(times), spatialResolution))
    psiL = np.zeros((len(times), spatialResolution))
    #dpsiR = np.zeros((len(times), spatialResolution))
    #dpsiL = np.zeros((len(times), spatialResolution))
    # spatialResolution of data
    sRh = len(hhh[0].psiL)
    # discretization of data
    hds = dInit.L / sRh
    length = sRh * hds
    ds = length / spatialResolution

    for (t, h) in enumerate(hhh):
        oldRange = [ni * hds for ni in range(sRh)]
        newRange = np.array([ds * i for i in range(spatialResolution)])
        f = interp1d(oldRange, h.psiL)
        psiL[t] = f(newRange)
        f = interp1d(oldRange, h.psiR)
        psiR[t] = f(newRange)
        #f = interp1d(oldRange, h.dpsiL)
        #dpsiL[t] = f(newRange)
        #f = interp1d(oldRange, h.dpsiR)
        #dpsiR[t] = f(newRange)

    # cartesian coordinates
    hR = [procedure.psi2path(psi, ds) for psi in psiR]
    hL = [procedure.psi2path(psi, ds) for psi in psiL]
    xflagR = np.array([h[0][cut:] + bx for h in hR])
    yflagR = np.array([h[1][cut:] + by for h in hR])
    xflagL = np.array([h[0][cut:] - bx for h in hL])
    yflagL = np.array([h[1][cut:] + by for h in hL])

    return {'body': body,
            'xflagR': xflagR,
            'yflagR': yflagR,
            'xflagL': xflagL,
            'yflagL': yflagL,
            'psiL': psiL,
            'psiR': psiR}


def noisyFlagellaAndBody(timeResolution=100,
                         spatialResolution=60,
                         omega=50,
                         noise=None):

    if noise is None:
        noise = np.ones((2, timeResolution + 1))
    assert len(noise[0]) == timeResolution + 1

    dt = 2 * np.pi / (omega * timeResolution)
    times = [i * dt for i in range(timeResolution + 1)]

    data = readData('/home/gary/MPI/bin/resistiveForce/bensMatlab/data')

    dInit = data2rftInit(data)

    # body is an ellipse
    body = dInit.body

    # where are the falgella docked?
    bx = dInit.bx
    by = dInit.by

    # tangent angles and normals in material frame
    hhh = [data2rftPsis(data, omega, time) for time in times]
    psiR = np.zeros((len(times), spatialResolution))
    psiL = np.zeros((len(times), spatialResolution))
    #dpsiR = np.zeros((len(times), spatialResolution))
    #dpsiL = np.zeros((len(times), spatialResolution))
    # spatialResolution of data
    sRh = len(hhh[0].psiL)
    # discretization of data
    hds = dInit.L / sRh
    length = sRh * hds
    ds = length / spatialResolution

    for (t, h) in enumerate(hhh):
        oldRange = [ni * hds for ni in range(sRh)]
        newRange = np.array([ds * i for i in range(spatialResolution)])
        f = interp1d(oldRange, h.psiL)
        psiL[t] = f(newRange)
        f = interp1d(oldRange, h.psiR)
        psiR[t] = f(newRange)
        #f = interp1d(oldRange, h.dpsiL)
        #dpsiL[t] = f(newRange)
        #f = interp1d(oldRange, h.dpsiR)
        #dpsiR[t] = f(newRange)

    # psi0
    psi0_L = np.mean(psiL, axis=0)
    psi0_R = np.mean(psiR, axis=0)
    psiL = psi0_L + ((psiL - psi0_L).transpose() * noise[0]).transpose()
    psiR = psi0_R + ((psiR - psi0_R).transpose() * noise[1]).transpose()

    # cartesian coordinates
    hR = [procedure.psi2path(psi, ds) for psi in psiR]
    hL = [procedure.psi2path(psi, ds) for psi in psiL]
    xflagR = np.array([h[0] + bx for h in hR])
    yflagR = np.array([h[1] + by for h in hR])
    xflagL = np.array([h[0] - bx for h in hL])
    yflagL = np.array([h[1] + by for h in hL])

    return {'body': body,
            'xflagR': xflagR,
            'yflagR': yflagR,
            'xflagL': xflagL,
            'yflagL': yflagL,
            'psiL': psiL,
            'psiR': psiR}


def amp_with_phi_092953(phi):
    """
    amplitude hase to be restricted such that there is no crossing of
    flagella with cell body.
    """
    phi_1 = 0.35 * 2 * np.pi
    phi_2 = 0.85 * 2 * np.pi
    phi_0 = 0.5 * (phi_1 + phi_2)
    f_0 = 0.97
    f_1 = 1.2
    coeff = 4 * (f_1 - f_0) / (phi_1 - phi_2) ** 2
    return min(coeff * (phi - phi_0) ** 2 + f_0, f_1)


def amp_phi_corrected_092953(amp, phi):
    """ correct the amplitude for steric interaction """
    return min(amp, amp_with_phi_092953(phi))


def ruloff_chlamy_092953(phi, amp, spatial_resolution=20, shortening=0.9):
    """
    that specific ruloff chlamy with already corrected amplitude, such that
    there are no crossings of surfaces.
    """

    def amp_with_phi(phi):
        """
        amplitude hase to be restricted such that there is no crossing of
        flagella with cell body.
        """
        phi_1 = 0.35 * 2 * np.pi
        phi_2 = 0.85 * 2 * np.pi
        phi_0 = 0.5 * (phi_1 + phi_2)
        f_0 = 0.97
        f_1 = 1.2
        coeff = 4 * (f_1 - f_0) / (phi_1 - phi_2) ** 2
        return min(coeff * (phi - phi_0) ** 2 + f_0, f_1)


    return ruloff_chlamy('20150602_092953_01', phi, min(amp, amp_with_phi(phi)),
                         spatial_resolution=spatial_resolution,
                         shortening=shortening)

