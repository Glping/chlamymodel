"""
This module provides mechanisms to create effective forces for basal flagellar
coupling. Crucial for the creation is the distance function, from which forces
are generated. The effective forces come from calibration of the cell for the
synchronously beating clamped cell. There, elastic forces are already
considered for ``p_l == p_r`` and ``a_l == a_r``.

Possible distance functions:
    - :func:`chlamymodel.basal.realistic_distance_function`
    - :func:`chlamymodel.basal.order_approximation`
    - :func:`chlamymodel.basal.distance_approximation`
    - :func:`chlamymodel.basal.easy_distance_approximation`
    - :func:`chlamymodel.basal.stretched_distance_approximation`

Pluggin those into :func:`chlamymodel.basal.generate_all_force_functions`
delivers effective forces, that can be used in
:class:`chlamymodel.seven.Driver`.
"""
import numpy as np
import FMM.chlamyinput as ci
import misc
import inspect


def get_basal_body_position(phi, amp):
    """ we assume the basal body to be at a similar position as the flagellar
    anchoring at the cell wall.

    :param phi: flagellar phase
    :param amp: flagellar amplitude
    """
    data = ci.ruloff_chlamy_092953(phi, amp, spatial_resolution=20)
    return ((data['xflagL'][1], data['yflagL'][1]),
            (data['xflagR'][1], data['yflagR'][1]))


def equilibrium_distance_function(distance_function):
    """ given a distance function that depends on the flagellar shape
    parameters, calculated the equilibrium distance as the phase average.

    :param distance_function: function with arguments p_l, a_l, p_r, a_r
    :return: the equilibrium distance (scalar)
    """
    return np.mean([distance_function(p, 1, p, 1) for p in np.linspace(0, 2 * np.pi, endpoint=False)])


def realistic_distance_function(other=False):
    """ get the distance function for realistic shapes.

    :param other: if ``other === False`` return the distance function for cell
        ``092953``, otherwise for ``063620``.
    :return: a tuple with the distance function taking the argument p_l, a_l,
        p_r, a_r and the equilibrium distance.
    """

    if not other:
        data = ci.ruloff_chlamy_092953(0, 1, spatial_resolution=20)
    else:
        data = ci.ruloff_chlamy('20150624_063620_01', 0, 1, 20)
    psi_func = data['psi_func']
    psis = [psi_func(0, p, 1) for p in np.linspace(0, 2 * np.pi, endpoint=False)]
    ds = data['length'] / 20

    dx = 2 * (data['dbx'] + np.sin(psis) * ds)
    b_0 = np.mean(dx)

    def realistic_distance(phi_l, amp_l, phi_r, amp_r):
        """ the actual distance function """
        ((x_l, y_l), (_, _)) = get_basal_body_position(phi_l, amp_l)
        ((_, _), (x_r, y_r)) = get_basal_body_position(phi_r, amp_r)
        x_l = -data['dbx'] - np.sin(psi_func(0, phi_l, amp_l)) * ds
        y_l = data['dby'] + np.cos(psi_func(0, phi_l, amp_l)) * ds
        x_r = data['dbx'] + np.sin(psi_func(0, phi_r, amp_r)) * ds
        y_r = data['dby'] + np.cos(psi_func(0, phi_r, amp_r)) * ds
        return np.sqrt((x_l - x_r) ** 2 + (y_l - y_r) ** 2)

    return (realistic_distance, b_0)


def order_approximation(second_order=1):
    """ vary the contribution of the second and third fourier mode to the
    flagellar shape

    :param second_order: coefficient for the importance of the second and third
        fourier mode. For ``second_order = 1``, we have full contribution, for
        ``second_order = 0``, there is no contribution.
    :return: distance function that depends on ``p_l, a_l, p_r, a_r``.
    """
    phis = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    (realistic_distance, equilibrium_distance) = realistic_distance_function()
    distance_function = Deriver(realistic_distance)
    coeffs = np.fft.rfft([distance_function(p, 1, 0, 1) for p in phis])

    equi = 1 / len(phis) * coeffs[0]

    def f(p):
        res = 1 / len(phis) * coeffs[0]
        res += 2 / len(phis) * coeffs[1] * np.exp(1j * p)
        res += second_order * 2 / len(phis) * coeffs[2] * np.exp(2j * p)
        res += second_order * 2 / len(phis) * coeffs[3] * np.exp(3j * p)
        return np.real(res)

    return (lambda p1, a1, p2, a2: 0.5 * (f(p1) + f(p2)), equi)


class Deriver:
    """ decorate a function with this class, makes it derivable

    :param func: a function of any number of parameters
    :param eps: the difference step for the difference equation

    :example:

    ::

        f = Deriver(lambda x, y, z: 2 * x ** 3 * y + z)
        f.d_d0.d_d0(1, 1, 0) # -> \\approx 12

    """
    def __init__(self, func, eps=0.0001):
        self.func = func
        self.eps = eps
        self.number_of_args = len(inspect.getargspec(self.func).args)

    def d_dn(self, n, *args):
        """ calculate the derivative with respect to the ``n`` th argument """
        new_args = list(args)
        a1 = new_args[n] - self.eps
        a2 = new_args[n] + self.eps
        new_args[n] = a1
        v1 = self.func(*new_args)
        new_args[n] = a2
        v2 = self.func(*new_args)
        return (v2 - v1) / (2 * self.eps)

    def __call__(self, *args):
        """ call the function """
        return self.func(*args)

    def __getattr__(self, name):
        """ any method with the name ``d_dn`` will return a new Deriver instance
        for the ``n`` th derivative of ``self`` """
        which = int(name.split('_')[1][1:])
        return Deriver(lambda *args: self.d_dn(which, *args))


def distance_approximation(p_l, a_l, p_r, a_r):
    """ generic distance function, periodic in ``p_l`` and ``p_r``, proportional
    to ``a_l`` and ``a_r``
    """
    return a_l * np.sin(p_l) + a_r * np.sin(p_r)


def stretched_distance_approximation(b_1):
    """ return a scaled version of :func:`chlamymodel.basal.distance_approximation` by
    ``b_1``"""
    return lambda a, b, c, d: distance_approximation(a, b, c, d) * b_1

def easy_distance_approximation(p_l, a_l, p_r, a_r):
    """ like :func:`chlamymodel.basal.distance_approximation`, but without the
    amplitude dependence """
    return np.sin(p_l) + np.sin(p_r)


def force(dist_func, dist_deriv, equilibrium_distance, stiffness=1):
    """ calculate the elastic force due to basal coupling

    :param dist_func: the distance function, dependent on four arguments
    :param dist_deriv: the derivative of which with respect to a respective
        variable
    :param equilibrium_distance: scalar value
    :param stiffness: the basal stiffness, a number
    :return: a function of ``p_l, a_l, p_r, a_r``
    """
    def inner(p_l, a_l, p_r, a_r):
        result = -stiffness * (dist_func(p_l, a_l, p_r, a_r) - equilibrium_distance)
        result *= dist_deriv(p_l, a_l, p_r, a_r)
        return result
    return inner


def generate_force_functions(distance, equilibrium_distance, stiffness=1):
    """ the elastic force function from basal coupling are computed here for
    each variabel.

    :param distance: distance function of four variables
    :param equilibrium_distance: scalar value
    :param stiffness: basal stiffness, a number
    :return: return a dictionary with keys, representing the shape variable, and
        values being the respective basal force.
    """

    force_func = {}
    for (i, v) in enumerate(['p_l', 'a_l', 'p_r', 'a_r']):
        force_func[v] = lambda *args, j=i: force(
            distance,
            lambda *bargs: distance.d_dn(j, *bargs),
            equilibrium_distance,
            stiffness=stiffness)(*args)

    return force_func


def generate_all_force_functions(distance_function,
                                 equilibrium_distance,
                                 equilibrium_distance_deviation,
                                 stiffness):
    """ the effective, elastic force function from basal coupling are computed
    here for each variable.

    :param distance_function: distance function of four variables
    :param equilibrium_distance: scalar value
    :param equilibrium_distance_deviation: scalar value, for calcium calculation
    :param stiffness: basal stiffness, a number
    :return: return a dictionary with keys, representing the shape variable, and
        values being the respective effective basal force.
    """

    force_funcs = generate_force_functions(
        distance_function, equilibrium_distance, stiffness)
    force_funcs_deviation = generate_force_functions(
        distance_function, equilibrium_distance_deviation, stiffness)


    def effective_force_left(force, force_deviated):
        def inner(p_l, a_l, p_r, a_r):
            f = force_deviated(p_l, a_l, p_r, a_r)
            return f - force(p_l, a_l, p_l, a_l)
        return inner

    def effective_force_right(force, force_deviated):
        def inner(p_l, a_l, p_r, a_r):
            f = force_deviated(p_l, a_l, p_r, a_r)
            return f - force(p_r, a_r, p_r, a_r)
        return inner

    return {'p_l': effective_force_left(force_funcs['p_l'], force_funcs_deviation['p_l']),
            'a_l': effective_force_left(force_funcs['a_l'], force_funcs_deviation['a_l']),
            'p_r': effective_force_right(force_funcs['p_r'], force_funcs_deviation['p_r']),
            'a_r': effective_force_right(force_funcs['a_r'], force_funcs_deviation['a_r'])}
