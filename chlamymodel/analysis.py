"""
This module provides two useful function:

    - :func:`chlamymodel.analysis.last_period` filters the provided
        :class:`chlamymodel.seven.Q`-data, returnong just the last period
    - :func:`chlamymodel.analysis.synchronization_strength_robust` allows us to
        compute the synchronozation strength from
        :class:`chlamymodel.seven.Q`-data
"""
import numpy as np
import misc


def last_period(q_data):
    """ assuming, ``q_data`` contains state dynamics that lasts longer than
    one period, the last period is returned.

    :param q_data: an array of shape (7, :) representing chlamydomonas state
        dynamics
    :return: the last period
    """
    phase_wrapped = q_data[0, :] % (2 * np.pi)
    jumps = np.where(np.diff(phase_wrapped) < 0)[0] + 1
    return q_data[:, jumps[-2]:jumps[-1]]


def synchronization_strength_from_qdata(q_data):
    """ Time dependent synchronization strength. The dependence on time
    can have three reasons: first, the relaxation of the dynamics due to
    elastic degrees of freedom did not happen, yet. Second, the small phase
    difference limit is not reached, yet. Third, the assumption of having
    two weakly interacting oscillator is not fulfilled.

    :param q_data: an array of shape (7, :) representing chlamydomonas state
        dynamics
    :return: a list of synchronization strengths as the mean of phase dependent
        strenghts within a cycle
    """
    phase_wrapped = q_data[0, :] % (2 * np.pi)
    phase_lag = np.abs(q_data[0, :] - q_data[1, :])
    jumps = np.where(np.diff(phase_wrapped) < 0)[0] + 1

    strengths = []
    for (i, j, k) in zip(jumps, jumps[1:], jumps[2:]):
        strength = np.mean([-np.log(pl_1 / pl_2)
                    for (pl_1, pl_2) in zip(phase_lag[j:k], phase_lag[i:j])])
        strengths.append(strength)

    return strengths


def first_minimum(data, blur_size=1):
    """ the first local minimum of a dataset is found. First, averaging within
    a window is done (convolution) for decreasing the effect of noise.

    :param data: a one-dimensional array
    :param blur_size: the size of the averaging window
    :return: the index of the first minimum
    """
    # first blur
    blurred = [np.mean(data[i-blur_size:i+blur_size+1])
               for i in range(len(data) - 2 * blur_size + 1)]

    for i in range(len(blurred) - 2):
        if blurred[i+1] < blurred[i] and blurred[i+1] < blurred[i+2]:
            return i + blur_size

    blurred = [(i, b) for (i, b) in enumerate(blurred) if not np.isnan(b)]
    return min(blurred, key=lambda h: h[1])[0] + blur_size


def choose_by_interval(strengths):
    """ We use a heuristic approach, for obtaining realistic values of the
    synchronization strength. The first values are wrong, due to relaxation
    dynamics and not fulfilled small phase difference assumption. The last
    values may be wrong due to numerical problems of small numbers.

    :param strengths: a list of synchronization strengths, from time dynamics
    :return: value of ``strengths`` with lowest derivative
    """
    if len(strengths) < 3:
        return (len(strengths) - 1, strengths[-1])
    length = int(len(strengths) / 3)
    intervals = [list(range(i, i + length)) for i in range(len(strengths) - length)]
    errors = []
    for interval in intervals:
        fit = misc.curve_fit(
            lambda t, l: l, interval, [strengths[i] for i in interval])
        errors.append(fit[1][0])
    minimum_index = int(first_minimum(errors) + length / 2)
    return (minimum_index, strengths[minimum_index])


def synchronization_strength_robust(q_data):
    """ a robust method for calculating the synchronization strength. It
    calculated for all times, and then the heuristic of
    :func:`chlamymodel.analysis.choose_by_interval` is used.

    :param q_data: chlamydomonas state dynamics
    :return: single value for the synchronization strength.
    """
    strengths = synchronization_strength_from_qdata(q_data)
    (index, strength) = choose_by_interval(strengths)
    return strength


def parameter_from_filename(filename):
    """ extract parameter from ``filename`` """
    return float(filename.split('_')[-1])

def sort_filenames(filenames):
    """ sort ``filenames`` by parameter """
    return sorted(filenames, key=parameter_from_filename)
