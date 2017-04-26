""" The heart of the chlamydomonas model. Here, the friction matrix is
pickled into a useful representation from hydrodynamic compution results
(:func:`chlamymodel.seven.pickle_friction`), or just read from a pickled file
(:func:`chlamymodel.seven.read_friction`). For this, all the components of the
friction matrix are fitted onto function with four arguments, representing the
shapes of the two flagella of chlamydomonas.

A variety of functions exist for manipulating the friction matrix according to
specific assumptions. For details, read the functions documentation. Examples
are
- :func:`chlamymodel.seven.change_viscosity`
- :func:`chlamymodel.seven.include_viscosity_and_efficiency`
- :func:`chlamymodel.seven.include_constant_phase_dissipation`
- :func:`chlamymodel.seven.include_efficiency_dissipation`
- :func:`chlamymodel.seven.include_internal_phase_dissipation`
- :func:`chlamymodel.seven.tune_hydrodynamic_interactions`
- :func:`chlamymodel.seven.turn_off_hydrodynamic_interactions`

The most central function of this module is
:func:`chlamymodel.seven.beat_with_forces`. In it, the time integration of the
chlamydomonas model happens. Inside it, the chlamydomonas state is read and
written, which is represented by an instance of class
:class:`chlamymodel.seven.Q`. The boundary conditions, such as constraints of
speeds or driving forces are represented by an instance of the
:class:`chlamymodel.seven.Driver` class.
"""
import numpy as np
# from numpy.fft import fft2, ifft2
# from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.interpolate import interp2d, interp1d
# import interpolation.interp2d
import copy


def pickle_friction():
    """ Create a pickled version of friction matrix data. This function can only
    be used, if the friction matrix for each particular state is saved in a
    respective file. See source code for details.
    """
    from pickle import load, dump
    filehandle = open('friction.pickle', 'wb')
    friction = np.array([[[[np.loadtxt(
        'FMBEM_simulations/friction-data/pl_{0:02d}_pr_{1:02d}_al_{2:02d}_ar_{3:02d}'.format(
            p_l, p_r, a_l, a_r))
                            for a_r in range(10)]
                           for a_l in range(10)]
                          for p_r in range(20)]
                         for p_l in range(20)])
    dump(friction, filehandle)
    filehandle.close()



def read_friction(flip_rotation=False, other_cell=False):
    """ Read the friction matrix in a pickled format. For this function to
    work, it is required that a folder ``.chlamymodel`` with the files
    ``symmetric_friction.pickle`` and symmetric_friction_other.pickle`` exists
    in the ``$HOME``-folder.

    :param flip_rotation: if ``True``, the rotational components flip sign
    :param other_cell: if ``False``, cell 092953 is used, otherwise cell 063620.
    :return: an array of shape (20, 20, 10, 10, 7, 7)
    """
    from pickle import load, dump
    import os.path
    if other_cell:
        # filehandle = open('/home/gary/.chlamymodel/symmetric_friction_other.pickle', 'rb')
        filehandle = open(os.path.expanduser(
            '~/.chlamymodel/symmetric_friction_other.pickle'), 'rb')
    else:
        # filehandle = open('/home/gary/.chlamymodel/symmetric_friction.pickle', 'rb')
        filehandle = open(os.path.expanduser(
            '~/.chlamymodel/symmetric_friction.pickle'), 'rb')
    #filehandle = open('friction.pickle', 'rb')
    friction = load(filehandle)
    filehandle.close()
    if flip_rotation:
        res = np.zeros([20, 20, 10, 10, 7, 7])
        for i in range(7):
            for j in range(7):
                if (i == 6 and j != 6) or (j == 6 and i != 6):
                    res[:, :, :, :, i, j] = -friction[:, :, :, :, i, j]
                else:
                    res[:, :, :, :, i, j] = friction[:, :, :, :, i, j]
        return res
    return friction



def fit_friction_component(component, border=5):
    """
    component is an array of shape (20, 20, 10, 10).
    fit a function onto the data.
    for each combination of the first two indices, calculate a paraboloid,
    resulting in 6 coefficients.
    Then calculate a cubic spline in the first two indices of each coefficient
    and return a function of four parameters.
    """

    N_A = 10
    domain_A = np.linspace(0.8, 1.2, N_A)
    N_phi = 20
    domain_phi = np.linspace(-2 * border * np.pi, 2 * np.pi * (border + 1 - 1 / N_phi), (2 * border + 1) * N_phi)
    coefficients_help = np.zeros((N_phi, N_phi, 6))
    for phi_l in range(N_phi):
        for phi_r in range(N_phi):

            A = np.ones((N_A ** 2, 6))
            G = np.zeros(N_A ** 2)

            for a_l in range(N_A):
                for a_r in range(N_A):
                    i = a_l * N_A + a_r
                    A[i, 1] = domain_A[a_l]
                    A[i, 2] = domain_A[a_r]
                    A[i, 3] = domain_A[a_l] ** 2
                    A[i, 4] = domain_A[a_r] ** 2
                    A[i, 5] = domain_A[a_l] * domain_A[a_r]
                    G[i] = component[phi_l, phi_r, a_l, a_r]

            c = np.linalg.lstsq(A, G)[0]
            #coefficients_help[phi_l, phi_r, :] = c
            coefficients_help[phi_r, phi_l, :] = c

    coefficients = [np.concatenate([coefficients_help[:, :, i]] * (2 * border + 1), axis=0)
            for i in range(6)]
    coefficients = [np.concatenate([coefficients[i]] * (2 * border + 1), axis=1)
            for i in range(6)]

    coeff_splines = [interp2d(domain_phi, domain_phi, coefficients[i],
                              kind='cubic')
            for i in range(6)]

    def fit_function(pl, pr, al, ar):
        ret = coeff_splines[0](pl, pr)
        ret += coeff_splines[1](pl, pr) * al
        ret += coeff_splines[2](pl, pr) * ar
        ret += coeff_splines[3](pl, pr) * al ** 2
        ret += coeff_splines[4](pl, pr) * ar ** 2
        ret += coeff_splines[5](pl, pr) * al * ar
        return ret[0]

    def amplitude_l_deriv(pl, pr, al, ar):
        ret = coeff_splines[1](pl, pr)
        ret += coeff_splines[5](pl, pr) * ar
        ret += coeff_splines[3](pl, pr) * al * 2
        return ret[0]

    def amplitude_r_deriv(pl, pr, al, ar):
        ret = coeff_splines[2](pl, pr)
        ret += coeff_splines[5](pl, pr) * al
        ret += coeff_splines[4](pl, pr) * ar * 2
        return ret[0]

    return (fit_function, amplitude_l_deriv, amplitude_r_deriv)



def fit_internal_dissipation_component(component, border=5):
    """
    component is a 2d field, periodic in its second variable.
    """
    component = np.array(component)
    (grid_a, grid_phi) = component.shape
    domain_A = np.linspace(0.8, 1.2, grid_a)
    domain_phi = np.linspace(-2 * border * np.pi, 2 * np.pi * (border + 1 - 1 / grid_phi), (2 * border + 1) * grid_phi)
    coefficients_help = np.zeros((grid_phi, 3))
    for phi in range(grid_phi):
        A = np.ones((grid_a, 3))
        G = np.zeros(grid_a)
        for a in range(grid_a):
            A[a, 1] = domain_A[a]
            A[a, 2] = domain_A[a] ** 2
            G[a] = component[a, phi]
        c = np.linalg.lstsq(A, G)[0]
        coefficients_help[phi, :] = c

    coefficients = [np.concatenate(
        [coefficients_help[:, i]] * (2 * border + 1))
            for i in range(3)]

    coeff_splines = [interp1d(domain_phi, coefficients[i], kind='cubic')
            for i in range(3)]

    def fit_function(p, a):
        ret = coeff_splines[0](p)
        ret += coeff_splines[1](p) * a
        ret += coeff_splines[2](p) * a ** 2
        return ret

    return fit_function





def read_friction_matrix(flip_rotation=False, modes=10, other_cell=False):
    """
    read friction matrices from FMBEM simulations and fit two dimensional
    functions for each component.
    """
    frictions = read_friction(flip_rotation=flip_rotation, other_cell=other_cell)
    res = [[fit_friction_component(frictions[:, :, :, :, i, j])#, border=4)
        for i in range(frictions.shape[4])]
        for j in range(frictions.shape[5])]
    return (np.array([[r[0] for r in re] for re in res]),
            np.array([[r[1] for r in re] for re in res]),
            np.array([[r[2] for r in re] for re in res]))


def restructure(matrix, indices):
    """
    matrix: cubic matrix.
    indices: all indices of the matrix in some order.
    return a matrix of same shape with indices shifted as first indices.

    e.g.:
    matrix = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    indices = [0, 2, 1]
    restructure(matrix, indices) == np.array([[0, 0, 0], [2, 2, 2], [1, 1, 1]])
    """
    # check input
    assert(len(set(indices)) == matrix.shape[0] == matrix.shape[1]
            and np.all(np.array(indices) < len(indices)))

    tmp = matrix[indices, :]
    return tmp[:, indices]


def backindices(indices):
    """
    given a permutation list, find the permutation list for converting back.

    e.g.:
    indices = [1, 0, 2]
    return [1, 0, 2]

    indices = [2, 0, 1]
    return [1, 2, 0]
    """
    l = len(indices)
    assert(len(set(indices)) == l
            and np.all(np.array(indices) < l))
    indices = np.array(indices)
    return [np.where(indices == i)[0][0] for i in range(l)]


def partially_invert_top_left(matrix, n):
    """
    partially invert matrix in its first n components.
    """
    g_pp = matrix[:n, :n]
    g_pp_inv = np.linalg.inv(g_pp)
    g_cc = matrix[n:, n:]
    g_cp = matrix[:n, n:]
    g_pc = matrix[n:, :n]

    ret = np.zeros(matrix.shape)
    ret[:n, :n] = g_pp_inv
    ret[n:, :n] = np.dot(g_pc, g_pp_inv)
    ret[:n, n:] = -np.dot(g_pp_inv, g_cp)
    ret[n:, n:] = -np.dot(g_pc, np.dot(g_pp_inv, g_cp)) + g_cc

    return ret



def friction_phi_from_position(phase, amplitude, omega=1, spatial_resolution=20):
    """
    internal friction should not only influence phase velocities but also the
    amplitude change. Friction is proportional to the curvature change.
    """
    epsilon_phi = 2 * np.pi / 1000000

    dphi_psi = psis_from_positions(
            phase + epsilon_phi, amplitude,
            omega=omega, spatial_resolution=spatial_resolution)
    dphi_psi -= psis_from_positions(
            phase - epsilon_phi, amplitude,
            omega=omega, spatial_resolution=spatial_resolution)
    dphi_psi /= 2 * epsilon_phi

    ds_dphi_psi = np.diff(dphi_psi) # divide by ds, later multiply

    return np.sum(ds_dphi_psi ** 2)

def friction_amp_from_position(phase, amplitude, omega=1, spatial_resolution=20):
    """
    internal friction should not only influence phase velocities but also the
    amplitude change. Friction is proportional to the curvature change.
    """
    epsilon_a = 0.4 / 10000

    da_psi = psis_from_positions(
            phase, amplitude + epsilon_a,
            omega=omega, spatial_resolution=spatial_resolution)
    da_psi -= psis_from_positions(
            phase, amplitude - epsilon_a,
            omega=omega, spatial_resolution=spatial_resolution)
    da_psi /= 2 * epsilon_a

    ds_da_psi = np.diff(da_psi) # divide by ds, later multiply

    return np.sum(ds_da_psi ** 2)


def change_viscosity(friction_func, viscosity=1):
    """
    I think that the friction matrix is proportional to the viscosity. Since
    I calculate all the friction matrix components with viscocity equal to 0.85,
    I can just divide the matrix by the wanted value.
    Higher viscosity, higher friction.
    """
    res = copy.deepcopy(friction_func)
    (n_i, n_j) = friction_func.shape

    def which(i, j):
        def func(a, b, c, d):
            return friction_func[i, j](a, b, c, d) / 0.85 * viscosity
        return func

    for i in range(n_i):
        for j in range(n_j):
            res[i, j] = which(i, j)

    return res


def include_viscosity_and_efficiency(
        friction_func, efficiency_ratio=0.2, viscosity=1):
    """
    viscosity is not just a multiplicative factor to the friction
    matrix if non-zero efficiency is assumed.
    """
    res = copy.deepcopy(friction_func)
    (n_i, n_j) = friction_func.shape


    def efficiency_which(i, j):

        def inner_efficient(a, b, c, d):
            ret = copy.deepcopy(friction_func[i, j])
            result = viscosity / 0.85 * ret(a, b, c, d)
            result += (1 - efficiency_ratio) / efficiency_ratio * ret(a, b, c, d)
            return result

        def inner(a, b, c, d):
            ret = copy.deepcopy(friction_func[i, j])
            return viscosity / 0.85 * ret(a, b, c, d)

        if (i, j) in [(0, 0), (1, 1), (2, 2), (3, 3),
                      (0, 2), (1, 3), (2, 0), (3, 1)]:
            return inner_efficient
        else:
            return inner

    for i in range(n_i):
        for j in range(n_j):
            res[i, j] = efficiency_which(i, j)

    return res



def include_constant_phase_dissipation(
        friction_func, efficiency_ratio=0.2):

    res = copy.deepcopy(friction_func)

    def which(i, j):
        g_i = (1 - efficiency_ratio) / (2 * np.pi * efficiency_ratio)
        grid = 100
        phis = np.linspace(0, 2 * np.pi * (1 - 1 / grid), grid)
        dphi = 2 * np.pi / grid
        g_i *= np.sum([friction_func[i, j](p, p, 1, 1) for p in phis]) * dphi
        def func(pl, pr, al, ar):
            return friction_func[i, j](pl, pr, al, ar) + g_i
        return func

    for i in [0, 1, 2, 3]:
        res[i, i] = which(i, i)
    res[0, 2] = which(0, 2)
    res[1, 3] = which(1, 3)
    res[2, 0] = which(2, 0)
    res[3, 1] = which(3, 1)

    return res


def include_efficiency_dissipation(
        friction_func, efficiency_ratio=0.2, amplitude_efficiency=True):
    res = copy.deepcopy(friction_func)

    def which(i, j):
        def func(pl, pr, al, ar):
            return friction_func[i, j](pl, pr, al, ar) / efficiency_ratio
        return func

    for i in [0, 1]:
        res[i, i] = which(i, i)
    if amplitude_efficiency:
        for i in [2, 3]:
            res[i, i] = which(i, i)
        res[0, 2] = which(0, 2)
        res[2, 0] = which(2, 0)
        res[1, 3] = which(1, 3)
        res[3, 1] = which(3, 1)

    return res


def include_internal_phase_dissipation(
        friction_func, dissipation_factor=10, omega=1):

    res = copy.deepcopy(friction_func)
    phis = np.linspace(0, 2 * np.pi * (1 - 1 / 20), 20)
    amps = np.linspace(0.8, 1.2, 10)

    int_phi = fit_internal_dissipation_component(
        np.array([[friction_phi_from_position(
            phi, amp, omega=omega, spatial_resolution=20)
                for phi in phis] for amp in amps]))
    int_amp = fit_internal_dissipation_component(
        np.array([[friction_amp_from_position(
            phi, amp, omega=omega, spatial_resolution=20)
                for phi in phis] for amp in amps]))

    def phi_l(pl, pr, al, ar):
        return friction_func[0, 0](pl, pr, al, ar) + int_phi(pl, al) * dissipation_factor
    res[0, 0] = phi_l
    def phi_r(pl, pr, al, ar):
        return friction_func[1, 1](pl, pr, al, ar) + int_phi(pr, ar) * dissipation_factor
    res[1, 1] = phi_r
    def amp_l(pl, pr, al, ar):
        return friction_func[2, 2](pl, pr, al, ar) + int_amp(pl, al) * dissipation_factor
    res[2, 2] = amp_l
    def amp_r(pl, pr, al, ar):
        return friction_func[3, 3](pl, pr, al, ar) + int_amp(pr, ar) * dissipation_factor
    res[3, 3] = amp_r

    return res


def tune_hydrodynamic_interactions(friction_func, fraction=1):

    """
    friction_func: 7x7 matrix containing function of 4 variables.
    set some components, representing hydrodynamic interactions between
    the two flagella to zero, which are amplitude and phase couplings.
    """
    res = copy.deepcopy(friction_func)
    res[0, 1] = lambda p1, p2, a1, a2: fraction * friction_func[0, 1](p1, p2, a1, a2)
    res[1, 0] = lambda p1, p2, a1, a2: fraction * friction_func[1, 0](p1, p2, a1, a2)
    res[1, 2] = lambda p1, p2, a1, a2: fraction * friction_func[1, 2](p1, p2, a1, a2)
    res[2, 1] = lambda p1, p2, a1, a2: fraction * friction_func[2, 1](p1, p2, a1, a2)
    res[2, 3] = lambda p1, p2, a1, a2: fraction * friction_func[2, 3](p1, p2, a1, a2)
    res[3, 2] = lambda p1, p2, a1, a2: fraction * friction_func[3, 2](p1, p2, a1, a2)
    res[0, 3] = lambda p1, p2, a1, a2: fraction * friction_func[0, 3](p1, p2, a1, a2)
    res[3, 0] = lambda p1, p2, a1, a2: fraction * friction_func[3, 0](p1, p2, a1, a2)
    res[0, 0] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[0, 0](p1, p1, a1, a1) + fraction * friction_func[0, 0](p1, p2, a1, a2)
    res[1, 1] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[1, 1](p2, p2, a2, a2) + fraction * friction_func[1, 1](p1, p2, a1, a2)
    res[2, 2] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[2, 2](p1, p1, a1, a1) + fraction * friction_func[2, 2](p1, p2, a1, a2)
    res[3, 3] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[3, 3](p2, p2, a2, a2) + fraction * friction_func[3, 3](p1, p2, a1, a2)
    res[1, 3] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[1, 3](p2, p2, a2, a2) + fraction * friction_func[1, 3](p1, p2, a1, a2)
    res[3, 1] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[3, 1](p2, p2, a2, a2) + fraction * friction_func[3, 1](p1, p2, a1, a2)
    res[0, 2] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[0, 2](p1, p1, a1, a1) + fraction * friction_func[0, 2](p1, p2, a1, a2)
    res[2, 0] = lambda p1, p2, a1, a2: (1 - fraction) * friction_func[2, 0](p1, p1, a1, a1) + fraction * friction_func[2, 0](p1, p2, a1, a2)
    return res


def turn_off_hydrodynamic_interactions(friction_func):
    """
    friction_func: 7x7 matrix containing function of 4 variables.
    set some components, representing hydrodynamic interactions between
    the two flagella to zero, which are amplitude and phase couplings.
    """
    res = copy.deepcopy(friction_func)
    res[0, 1] = lambda p1, p2, a1, a2: 0
    res[1, 0] = lambda p1, p2, a1, a2: 0
    res[1, 2] = lambda p1, p2, a1, a2: 0
    res[2, 1] = lambda p1, p2, a1, a2: 0
    res[2, 3] = lambda p1, p2, a1, a2: 0
    res[3, 2] = lambda p1, p2, a1, a2: 0
    res[0, 3] = lambda p1, p2, a1, a2: 0
    res[3, 0] = lambda p1, p2, a1, a2: 0

    def which(i, j, left=True):
        def func(p1, p2, a1, a2):
            if left:
                return friction_func[i, j](p1, p1, a1, a1)
            else:
                return friction_func[i, j](p2, p2, a2, a2)
        return func

    res[0, 0] = which(0, 0, left=True)
    res[1, 1] = which(1, 1, left=False)
    res[2, 2] = which(2, 2, left=True)
    res[3, 3] = which(3, 3, left=False)

    return res


def partially_invert(matrix, indices):
    """ partially invert a matrix in the given indices

    :param matrix: rectangular matrix.
    :param indices: list of indices.
    :return: partially inverted matrix in the indices given.
    """
    assert(matrix.shape[0] == matrix.shape[1])
    # restructure
    n = matrix.shape[0]
    allindices = copy.deepcopy(indices)
    for i in range(n):
        if i not in indices:
            allindices.append(i)
    restructured = restructure(matrix, allindices)
    # partially invert
    part_invert = partially_invert_top_left(restructured, len(indices))
    # restructure back
    return restructure(part_invert, backindices(allindices))


def effective_system_force_free(friction):
    g_11 = friction[:4, :4]
    g_12 = friction[:4, 4:]
    g_21 = friction[4:, :4]
    g_22 = friction[4:, 4:]
    return g_11 - np.dot(g_12, np.dot(np.linalg.inv(g_22), g_21))


def effective_system_dynamics_free(friction):
    return friction[:4, :4]


def function_matrix_at(matrix, u, v, w, x):
    """
    matrix: two dimensional array, that contains functions of two variables.
    x, y: coordinates at which the functions are to be evaluated.
    return a matrix, containing scalar components.
    """
    (dx, dy) = matrix.shape
    return np.array([[matrix[j, i](u, v, w, x)
                        for i in range(dy)]
                        for j in range(dx)]).reshape((dx, dy))


def amplitude_stability(phi, gamma, omega=2*np.pi*0.05, effection=effective_system_force_free):

    def obtain_four_dof(amp):
        friction = function_matrix_at(gamma, phi, phi, amp, amp)
        return effection(friction)

    four_dof_low = obtain_four_dof(0.999999)
    four_dof = obtain_four_dof(1)
    four_dof_high = obtain_four_dof(1.000001)

    g_pp = four_dof[0, 0] + four_dof[0, 1]
    g_ap = four_dof[2, 0] + four_dof[2, 1]
    g_pa = four_dof[0, 2] + four_dof[0, 3]
    g_aa = four_dof[2, 2] + four_dof[2, 3]
    g_ap_high = four_dof_high[2, 0] + four_dof_high[2, 1]
    g_ap_low = four_dof_low[2, 0] + four_dof_low[2, 1]
    g_pp_high = four_dof_high[0, 0] + four_dof_high[0, 1]
    g_pp_low = four_dof_low[0, 0] + four_dof_low[0, 1]

    dg_ap = (g_ap_high - g_ap_low) / 0.000002
    dg_pp = (g_pp_high - g_pp_low) / 0.000002

    tauterm = g_aa - g_ap / g_pp * g_pa
    omegaterm = dg_ap - g_ap / g_pp * dg_pp

    return omega * omegaterm / tauterm


def stiffness(phi, gamma, tau_A=5.9, omega=2*np.pi*0.05, effection=effective_system_force_free):

    def obtain_four_dof(amp):
        friction = function_matrix_at(gamma, phi, phi, amp, amp)
        return effection(friction)

    four_dof_low = obtain_four_dof(0.999999)
    four_dof = obtain_four_dof(1)
    four_dof_high = obtain_four_dof(1.000001)

    g_pp = four_dof[0, 0] + four_dof[0, 1]
    g_ap = four_dof[2, 0] + four_dof[2, 1]
    g_pa = four_dof[0, 2] + four_dof[0, 3]
    g_aa = four_dof[2, 2] + four_dof[2, 3]
    g_ap_high = four_dof_high[2, 0] + four_dof_high[2, 1]
    g_ap_low = four_dof_low[2, 0] + four_dof_low[2, 1]
    g_pp_high = four_dof_high[0, 0] + four_dof_high[0, 1]
    g_pp_low = four_dof_low[0, 0] + four_dof_low[0, 1]

    dg_ap = (g_ap_high - g_ap_low) / 0.000002
    dg_pp = (g_pp_high - g_pp_low) / 0.000002

    tauterm = g_aa - g_ap / g_pp * g_pa
    omegaterm = dg_ap - g_ap / g_pp * dg_pp

    return tauterm / tau_A - omega * omegaterm


def bordered_stiffness(phi, amp, q_amp, friction_func, tau_A, omega,
                       effection=effective_system_dynamics_free):

    def func_a(x, a=0.01, b=100):
        return a * (np.exp(-b * (1.21 - x)))# / abs(1.21 - x)

    def func_b(x, a=1, b=100):
        if x < 1:
            return -func_a(2 - x, a=a, b=b)
        else:
            return func_a(x, a=a, b=b)

    res = q_amp(phi)
    if callable(tau_A):
        res -= stiffness(
            phi, friction_func, tau_A=tau_A(phi), omega=omega,
            effection=effection) * (amp - 1)
    else:
        res -= stiffness(
            phi, friction_func, tau_A=tau_A, omega=omega,
            effection=effection) * (amp - 1)
    res -= func_b(amp, a=1000, b=100)
    return res


def basal_body_force_left(p_l, p_r, k_b=0):
    # res = np.sin(0.5 * (p_l + p_r)) * np.cos(0.5 * (p_l - p_r))
    # return -k_b * (res - np.sin(p_l))
    return -k_b * (np.sin(p_r) - np.sin(p_l)) * np.cos(p_l)


def basal_body_force_right(p_l, p_r, k_b=0):
    return -k_b * (np.sin(p_l) - np.sin(p_r)) * np.cos(p_r)



class Q(object):
    """ An instance of ``Q`` represent the state of state dynamics of a
    chlamydomonas. There are three main purposes.

    - scalar values may be stored -> a state
    - lists or arrays may be stored -> a state dynamics
        If this is the case, then the ``Q`` instance is indexable, and a
        scalar ``Q`` instance is returned.
    - function may be stored -> a prescribed dynamics or prescribed conjugate
        forces. If this is the case, then the ``Q`` instance is callable.

    :param coordinates: an iterable with seven elements, representing the state
        of chlamydomonas.
    """

    def __init__(self, coordinates):
        self.phase_left = coordinates[0]
        self.phase_right = coordinates[1]
        self.amplitude_left = coordinates[2]
        self.amplitude_right = coordinates[3]
        self.x = coordinates[4]
        self.y = coordinates[5]
        self.orientation = coordinates[6]

    def map(self, f):
        """ apply a function to all elements, and return a new ``Q`` object.

        :param f: the function to be applied.
        """
        return Q([f(self.phase_left),
                  f(self.phase_right),
                  f(self.amplitude_left),
                  f(self.amplitude_right),
                  f(self.x),
                  f(self.y),
                  f(self.orientation)])

    def append(self, qscalar):
        """ assuming that the current ``Q`` instance contain lists, append each
        state variable to ``self``.

        :param qscalar: instance of ``Q``, containing scalars.
        :return: modifies qlist and returns it.
        """
        self.phase_left.append(qscalar.phase_left)
        self.phase_right.append(qscalar.phase_right)
        self.amplitude_left.append(qscalar.amplitude_left)
        self.amplitude_right.append(qscalar.amplitude_right)
        self.x.append(qscalar.x)
        self.y.append(qscalar.y)
        self.orientation.append(qscalar.orientation)
        return self

    def __getitem__(self, i):
        """
        indexing is possible if the state variables are indexable, which is true
        for lists or arrays.

        :param i: integer that represents an index
        """
        return Q([self.phase_left[i],
                  self.phase_right[i],
                  self.amplitude_left[i],
                  self.amplitude_right[i],
                  self.x[i],
                  self.y[i],
                  self.orientation[i]])

    def __len__(self):
        """ the length of a ``Q`` dynamics is the number of steps available """
        return len(self.phase_left)

    def init(self):
        """
        convenience function returning a ``Q``-dynamics without the last time
        step.
        """
        return self.map(lambda ll: ll[:-1])

    def __call__(self, t, q):
        """ to be able to call the ``Q`` instance, the variables must contain
        functions.

        :param t: a time argument for ``x`` and ``y``
        :param q: a ``Q`` argument for the other state variable functions
        :return: a ``Q`` object of functions return values
        """
        return Q([self.phase_left(q),
                  self.phase_right(q),
                  self.amplitude_left(q),
                  self.amplitude_right(q),
                  self.x(t),
                  self.y(t),
                  self.orientation(q)])

    def to_list(self):
        """ return self as a list, as an argument to ``Q`` instantiation would
        look like """
        return [self.phase_left, self.phase_right,
                self.amplitude_left, self.amplitude_right,
                self.x, self.y, self.orientation]

    def to_array(self):
        """ return numpy array of ``self.to_list``'s return value """
        return np.array(self.to_list())

    def __neg__(self):
        """ return ``Q`` object with elements negated """
        return self.map(lambda q: -q)

    def __add__(self, other):
        """ add two ``Q`` instances

        :param other: ``Q`` object.
        :return: ``Q`` object.
        """
        return self.map2(other, lambda a, b: a + b)

    def __rmul__(self, scalar):
        """ multiplication with a scalar

        :param scalar: a scalar.
        :return: a ``Q`` object with each field multiplied by scalar.
        """
        return self.map(lambda q: q * scalar)

    def map2(self, other, f):
        """ apply a binary operation on each state variable

        :param other: a ``Q`` object
        :param f: a function that takes to arguments and returns anything
        :return: a ``Q`` object
        """
        return Q([f(self.phase_left, other.phase_left),
                  f(self.phase_right, other.phase_right),
                  f(self.amplitude_left, other.amplitude_left),
                  f(self.amplitude_right, other.amplitude_right),
                  f(self.x, other.x),
                  f(self.y, other.y),
                  f(self.orientation, other.orientation)])

    def __mul__(self, other):
        """ multiply two ``Q`` objects, property wise.

        :param other: ``Q`` object
        :return: ``Q`` object
        """
        return self.map2(other, lambda x, y: x * y)

    def __repr__(self):
        """ generate a string for the purpose of debugging

        :return: a string
        """
        res = 'phase left:\n{0}\nphase right:\n{1}\n'.format(
            self.phase_left, self.phase_right)
        res += 'amplitude left:\n{0}\namplitude right:\n{1}\n'.format(
            self.amplitude_left, self.amplitude_right)
        res += 'x:\n{0}\ny:\n{1}\norientation:\n{2}\n'.format(
            self.x, self.y, self.orientation)
        return res


class Driver(object):
    """ The ``Driver`` instance represents the prescription of the dynamics
    of the chlamydomonas model. With it, we state if the cell is clamped or not.

    :param types: instance of ``Q``, containing boolean.
        If a state variable is True, then a velocity dynamics is prescribed, if
        it is False, then a force dynamics is prescribed.
    :param functions: instance of ``Q``, containing functions of time for ``x``
        and ``y`` and functions of a ``Q`` instance for the other variables

    :example:

    ::

        # a freely swimming cell, with presribed phase and amplitude dynamics
        free_swimming_driver = Driver(
            Q([True, True, True, True, False, False, False]),
            Q([lambda q: 1, lambda q: 1, lambda q: 0, lambda q: 0,
               lambda t: 0, lambda t: 0, lambda q: 0]))

        # and the respective clamped cell
        clamped_driver = Driver(
            Q([True, True, True, True, True, True, True]),
            Q([lambda q: 1, lambda q: 1, lambda q: 0, lambda q: 0,
               lambda t: 0, lambda t: 0, lambda q: 0]))

    For more examples on how to build more sophisticated drivers, visit the
    module :class:`chlamymodel.drivers`.
    """

    def __init__(self, types, functions):
        """
        types: instance of Q, containing boolean. True <=> velocity,
                                                  False <=> force
        functions: instance of Q, containing function of time.
        """
        self.types = types
        self.functions = functions

    def demix(self, q):
        """ containing a vector of mixed quantities (velocities and forces),
        demix it together with a vector of the inverse quantities.

        :param q: a q object, containing the inverse quantities of self
        :return: two q objects in a tuple:
            (trueish (velocity), falsish (force) ones)
        """
        qtrue = []
        qfalse= []

        if self.types.phase_left:
            qtrue.append(self.functions.phase_left)
            qfalse.append(q.phase_left)
        else:
            qfalse.append(self.functions.phase_left)
            qtrue.append(q.phase_left)
        if self.types.phase_right:
            qtrue.append(self.functions.phase_right)
            qfalse.append(q.phase_right)
        else:
            qfalse.append(self.functions.phase_right)
            qtrue.append(q.phase_right)

        if self.types.amplitude_left:
            qtrue.append(self.functions.amplitude_left)
            qfalse.append(q.amplitude_left)
        else:
            qfalse.append(self.functions.amplitude_left)
            qtrue.append(q.amplitude_left)
        if self.types.amplitude_right:
            qtrue.append(self.functions.amplitude_right)
            qfalse.append(q.amplitude_right)
        else:
            qfalse.append(self.functions.amplitude_right)
            qtrue.append(q.amplitude_right)

        if self.types.x:
            qtrue.append(self.functions.x)
            qfalse.append(q.x)
        else:
            qfalse.append(self.functions.x)
            qtrue.append(q.x)
        if self.types.y:
            qtrue.append(self.functions.y)
            qfalse.append(q.y)
        else:
            qfalse.append(self.functions.y)
            qtrue.append(q.y)
        if self.types.orientation:
            qtrue.append(self.functions.orientation)
            qfalse.append(q.orientation)
        else:
            qfalse.append(self.functions.orientation)
            qtrue.append(q.orientation)

        return (Q(qtrue), Q(qfalse))


    def partially_invert_indices(self):
        """
        :return: the indices, that can be used by
            :func:`chlamymodel.seven.partially_invert`
        """
        res = []
        if self.types.phase_left: res.append(0)
        if self.types.phase_right: res.append(1)
        if self.types.amplitude_left: res.append(2)
        if self.types.amplitude_right: res.append(3)
        if self.types.x: res.append(4)
        if self.types.y: res.append(5)
        if self.types.orientation: res.append(6)
        return res

    def __call__(self, t, q):
        """ call the function vector

        :param t: scalar, representing time
        :param q: a :class:`chlamymodel.seven.Q` vector
        :return: a :class:`chlamymodel.seven.Q` object.
        """
        return self.functions(t, q)

    def relate(self, relator, time, q):
        """ solve system of equations

        :param relator: the force velocity relation. It is a partially inverted
            friction matrix, see also :func:`chlamymodel.seven.partially_invert`
        :return: forces and velocities.
        """
        mixed = self(time, q)
        solution = Driver(self.types.map(lambda x: not x),
                          Q(np.linalg.solve(relator, mixed.to_list())))
                          #Q(np.linalg.lstsq(relator, mixed.to_list())[0]))
        return solution.demix(mixed)


def fit_strength(positions, period, dt, offset=3):
    """
    positions: Q-object, containing lists.
    period: period of the freely swimming cell.
    return the synchronization strength.
    """
    from misc import wrap
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d

    phase_lag = np.abs(np.array(positions.phase_left) - np.array(positions.phase_right))
    f_phase = interp1d(
            [dt * i for i in range(len(positions))],
            positions.phase_left, kind='cubic')
    f_phase_lag = interp1d(
            [dt * i for i in range(len(positions))],
            phase_lag, kind='cubic')
    ts = np.linspace(0, (len(positions) - 1) * dt, 10000)
    phase = f_phase(ts)
    phase_lag = np.log(f_phase_lag(ts))

    indices = np.where(np.diff(wrap(phase)) < 0)

    template = lambda t, a, l: a - l * t / period
    xx = [i * dt * len(positions) / 10000 for i in indices[0][offset:]]
    yy = phase_lag[indices[0][offset:]]
    try:
        (params, errs) = curve_fit(template, xx, yy)
    except:
        return None

    return (params[1], errs[1, 1])


def get_strength(positions):
    """
    get synchronization strength from different method than fit_strength.
    Use only the last two phi=0 points to obtain the synchronization strength
    and the respective phase lag.
    """
    from misc import wrap
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    phase_lag = np.array(positions.phase_left) - np.array(positions.phase_right)
    f_phase = interp1d(
            list(range(len(positions))),
            positions.phase_left, kind='cubic')
    f_phase_lag = interp1d(
            list(range(len(positions))),
            phase_lag, kind='cubic')
    ts = np.linspace(0, (len(positions) - 1), 10000)
    phase = f_phase(ts)
    phase_lag = f_phase_lag(ts)

    indices = np.where(np.diff(wrap(phase)) < 0)

    strength = phase_lag[indices[0][-2]] - phase_lag[indices[0][-1]]
    strength /= phase_lag[indices[0][-2]]
    return (phase_lag[indices[0][-2]] % (2 * np.pi), strength)


def rotate_friction(friction, angle):
    """
    friction is a n_q x n_q matrix. The components representing cartesian
    velocity are to be rotated, which are components 2 and 3.
    """
    generalized_rot = np.zeros((7, 7))
    generalized_rot_inv = np.zeros((7, 7))

    for i in [0, 1, 2, 3, 6]:
        generalized_rot[i, i] = 1
        generalized_rot_inv[i, i] = 1

    generalized_rot[4:6, 4:6] = np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
    generalized_rot_inv[4:6, 4:6] = np.array([[np.cos(-angle), -np.sin(-angle)],
                                              [np.sin(-angle), np.cos(-angle)]])

    return np.dot(generalized_rot, np.dot(friction, generalized_rot_inv))


def wrapQ(q):
    """ make phase of q object be in between 0 and 2\pi """
    return Q([q.phase_left % (2 * np.pi), q.phase_right % (2 * np.pi),
             q.amplitude_left, q.amplitude_right,
             q.x, q.y, q.orientation])




def beat_ode_without_rotation(
    initial_conditions, driver, dt, number_of_steps,
    friction_func=None, callback=lambda t, p, v, f:None,
    symmetric=False, backward=False, amplitude_only=False):
    """
    initial_conditions: instance of Q, containing scalars.
    driver: instance of Driver.
    dt: timestep duraction.
    number_of_steps: number of steps to be simulated.
    friction_func: a 7x7 matrix containing functions of phase and amplitude
    callback: function taking time, position, velocity and force as input.

    return: three Q-objects, containing lists:
    - trajectory
    - velocity
    - force
    """
    from scipy.integrate import odeint
    if friction_func is None:
        friction_func = read_friction_matrix()[0]

    current_state = initial_conditions
    #trajectory.append(current_state)

    def one_step(current_state, time):

        current_state = Q(current_state)

        phi_left = current_state.phase_left % (2 * np.pi)
        phi_right = current_state.phase_right % (2 * np.pi)

        friction_evaluated = function_matrix_at(friction_func,
                                                phi_left,
                                                phi_right,
                                                current_state.amplitude_left,
                                                current_state.amplitude_right)

        relator = partially_invert(friction_evaluated,
                                   driver.partially_invert_indices())

        (v, f) = driver.relate(relator, time, wrapQ(current_state))

        if symmetric:
            phase_v_mean = 0.5 * (v.phase_left + v.phase_right)
            v.phase_left = phase_v_mean
            v.phase_right = phase_v_mean
            amplitude_v_mean = 0.5 * (v.amplitude_left + v.amplitude_right)
            v.amplitude_left = amplitude_v_mean
            v.amplitude_right = amplitude_v_mean
        if backward:
            v = -v
        if amplitude_only:
            v.phase_left = 0
            v.phase_right = 0

        callback(time, current_state, v, f)

        return v.to_list()

    return Q(odeint(one_step, current_state.to_list(),
                    np.linspace(0, (number_of_steps - 1) * dt, number_of_steps),
                    #ixpr=True,
                    atol=10**(-12),
                    rtol=10**(-12)).transpose())

def beat_ode(initial_conditions, driver, dt, number_of_steps,
    friction_func=None, callback=lambda t, p, v, f:None,
    symmetric=False, backward=False):
    """
    initial_conditions: instance of Q, containing scalars.
    driver: instance of Driver.
    dt: timestep duraction.
    number_of_steps: number of steps to be simulated.
    friction_func: a 7x7 matrix containing functions of phase and amplitude
    callback: function taking time, position, velocity and force as input.

    return: three Q-objects, containing lists:
    - trajectory
    - velocity
    - force
    """
    from scipy.integrate import odeint
    if friction_func is None:
        friction_func = read_friction_matrix()[0]

    current_state = initial_conditions
    #trajectory.append(current_state)

    def one_step(current_state, time):

        current_state = Q(current_state)

        phi_left = current_state.phase_left % (2 * np.pi)
        phi_right = current_state.phase_right % (2 * np.pi)

        friction_evaluated = function_matrix_at(friction_func,
                                                phi_left,
                                                phi_right,
                                                current_state.amplitude_left,
                                                current_state.amplitude_right)

        friction_rotated = rotate_friction(friction_evaluated,
                                           current_state.orientation)

        relator = partially_invert(friction_rotated,
                                   driver.partially_invert_indices())

        (v, f) = driver.relate(relator, time, wrapQ(current_state))

        if symmetric:
            phase_v_mean = 0.5 * (v.phase_left + v.phase_right)
            v.phase_left = phase_v_mean
            v.phase_right = phase_v_mean
            amplitude_v_mean = 0.5 * (v.amplitude_left + v.amplitude_right)
            v.amplitude_left = amplitude_v_mean
            v.amplitude_right = amplitude_v_mean
        if backward:
            v = -v

        callback(time, current_state, v, f)

        return v.to_list()

    return Q(odeint(one_step, current_state.to_list(),
                    np.linspace(0, (number_of_steps - 1) * dt, number_of_steps),
                    ixpr=True,
                    atol=10**(-12),
                    rtol=10**(-12)).transpose())



def Qlist():
    return Q([[], [], [], [], [], [], []])


def rebeat_positions(positions, driver, dt, number_of_steps,
             friction_func=None):
    """
    positions: instance of Q, containing lists.
    driver: instance of Driver.
    dt: timestep duraction.
    number_of_steps: number of steps to be simulated.

    return: three Q-objects, containing lists:
    - trajectory
    - velocity
    - force
    """
    from scipy.integrate import odeint
    if friction_func is None:
        friction_func = read_friction_matrix()[0]

    velocities = Qlist()
    forces = Qlist()

    def one_step(current_state, time):
        nonlocal velocities
        nonlocal forces

        phi_left = current_state.phase_left % (2 * np.pi)
        phi_right = current_state.phase_right % (2 * np.pi)

        friction_evaluated = function_matrix_at(friction_func,
                                                phi_left,
                                                phi_right,
                                                current_state.amplitude_left,
                                                current_state.amplitude_right)

        friction_rotated = rotate_friction(friction_evaluated,
                                           current_state.orientation)

        relator = partially_invert(friction_rotated,
                                   driver.partially_invert_indices())

        (v, f) = driver.relate(relator, time, wrapQ(current_state))
        velocities.append(v)
        forces.append(f)

    [one_step(positions[i], dt * i) for i in range(number_of_steps)]
    return (velocities, forces)



def velocities_from_positions(positions, dt):
    velocities = positions.map(lambda ll: np.diff(np.array(ll)) / dt)
    return velocities


def forces_from_positions(positions, dt, friction_func=None):

    if friction_func is None:
        friction_func = read_friction_matrix()[0]
    velocities = velocities_from_positions(positions, dt)
    forces = Qlist()
    for i in range(len(velocities)):

        current_state = positions[i]
        phi_left = current_state.phase_left % (2 * np.pi)
        phi_right = current_state.phase_right % (2 * np.pi)

        friction_evaluated = function_matrix_at(friction_func,
                                                phi_left, phi_right,
                                                current_state.amplitude_left,
                                                current_state.amplitude_right)

        friction_rotated = rotate_friction(friction_evaluated,
                                           current_state.orientation)

        forces.append(Q(np.dot(friction_rotated, velocities[i].to_list())))

    return forces


# def beat_with_velocities(initial_conditions, driver, dt, number_of_steps,
#                          friction_func=None):
#     if friction_func is None:
#         friction_func = read_friction_matrix()[0]
#     positions = beat_ode(
#             initial_conditions, driver,
#             dt, number_of_steps, friction_func)
#     velocities = velocities_from_positions(positions, dt)
#     return (positions.init(), velocities)


def beat_with_forces(initial_conditions, driver, dt, number_of_steps,
                     friction_func=None, callback=lambda a, b, c, d: 0,
                     symmetric=False, backward=False, amplitude_only=False):

    if friction_func is None:
        friction_func = read_friction_matrix()[0]

    positions = beat_ode_without_rotation(
            initial_conditions, driver,
            dt, number_of_steps, friction_func,
            callback=callback, symmetric=symmetric,
            backward=backward, amplitude_only=amplitude_only)
    (velocities, forces) = rebeat_positions(
            positions, driver, dt, number_of_steps, friction_func)

    return (positions, velocities, forces)


def beat(initial_conditions, driver, dt, number_of_steps,
         friction_func=None):
    """
    initial_conditions: instance of Q, containing scalars.
    driver: instance of Driver.
    dt: timestep duraction.
    number_of_steps: number of steps to be simulated.

    return: three Q-objects, containing lists:
    - trajectory
    - velocity
    - force
    """
    if friction_func is None:
        friction_func = read_friction_matrix()[0]

    trajectory = Qlist()
    velocity = Qlist()
    force = Qlist()

    current_state = initial_conditions
    #trajectory.append(current_state)

    def one_step(time):

        nonlocal trajectory
        nonlocal velocity
        nonlocal force
        nonlocal current_state

        phi_left = current_state.phase_left % (2 * np.pi)
        phi_right = current_state.phase_right % (2 * np.pi)

        friction_evaluated = function_matrix_at(friction_func,
                                                phi_left,
                                                phi_right,
                                                current_state.amplitude_left,
                                                current_state.amplitude_right)

        friction_rotated = rotate_friction(friction_evaluated,
                                           current_state.orientation)

        relator = partially_invert(friction_rotated,
                                   driver.partially_invert_indices())

        (v, f) = driver.relate(relator, time, wrapQ(current_state))

        velocity.append(v)
        force.append(f)
        current_state = current_state + dt * v
        trajectory.append(current_state)

    [one_step(dt * i) for i in range(number_of_steps)]
    return (trajectory, velocity, force)


def const(value):
    """ return a constant function """
    return lambda *t: value


def read_phi_func_2(filename, modes=20):
    """
    read the file filename and fit a 2\pi periodic function on the data
    """
    import interpolation.interp1d
    data = np.loadtxt(filename)
    grid = data.shape[0]
    return interpolation.interp1d.periodic(data, modes=modes)

def read_phi_func(filename, border=3):
    """
    read the file filename and fit a 2\pi periodic function on the data
    """
    from scipy.interpolate import interp1d
    data = np.loadtxt(filename)
    grid = data.shape[0]
    phis = np.linspace(
            -border * 2 * np.pi / grid,
            2 * np.pi * (1 + (border - 1) / grid),
            grid + 2 * border)
    return interp1d(phis, np.concatenate(
        [data[-border:], data, data[:border]]),
        kind='cubic')


def read_phi_func_3(filename, border=4):
    """
    read the file filename and fit a 2\pi periodic function on the data
    """
    data = np.loadtxt(filename)
    return make_phi_func(data, border=border)


def make_phi_func(data, border=4):
    from scipy.interpolate import interp1d
    data = np.array(data)
    grid = data.shape[0]
    phis = np.linspace(
            -border * 2 * np.pi,
            2 * np.pi * (border + 1 - 1 / grid),
            (2 * border + 1) * grid)
    return interp1d(phis, np.concatenate(
        [data] * (2 * border + 1)),
        kind='cubic')


def phi_func_to_t_func(f, period):
    """
    convert a phase dependent function to a time dependent function.
    """
    g = lambda t: f((t / period * 2 * np.pi) % (2 * np.pi))
    return g


def load_phi_force_functions(border=2):
    q_phi_l = read_phi_func_3('phase_force_left')#, border=border)
    q_phi_r = read_phi_func_3('phase_force_right')#, border=border)
    return (q_phi_l, q_phi_r)


def load_amp_force_functions(border=2):
    q_amp_l = read_phi_func_3('amplitude_force_left')#, border=border)
    q_amp_r = read_phi_func_3('amplitude_force_right')#, border=border)
    return (q_amp_l, q_amp_r)


def load_force_functions(border=1):
    (q_phi_l, q_phi_r) = load_phi_force_functions(border=border)
    (q_amp_l, q_amp_r) = load_amp_force_functions(border=border)
    return (q_phi_l, q_phi_l, q_amp_l, q_amp_l)

def load_forces():
    q_phi_l = read_phi_func('phase_force_left')
    q_amp_l = read_phi_func('amplitude_force_left')
    return (q_phi_l, q_amp_l)


def save_force_data(q_phi_l, q_phi_r, q_amp_l, q_amp_r):
    np.savetxt('phase_force_left', q_phi_l)
    np.savetxt('phase_force_right', q_phi_r)
    np.savetxt('amplitude_force_left', q_amp_l)
    np.savetxt('amplitude_force_right', q_amp_r)
