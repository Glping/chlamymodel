from copy import deepcopy
import numpy as np
import itertools


class Binner:
    """ A binner is a data structure, holding orderable values. Values
    are added on the fly and immediately sorted into the appropriate
    bin. A bin consists of an interval and an array inside it. Functional
    access to the data is provided.

    :param bins: number of bins
    :param range_: range, tuple of minimum and maximum

    :example:

    ::

        b = Binner(2, (0, 10))
        b.append(1)
        b.append(6)
        b.append(2)
        b.map(lambda l: len(l)) # -> [2, 1]

    """

    def __init__(self, bins, range_):
        borders = np.linspace(range_[0], range_[1], num=bins+1)
        self._data = {b: [] for b in zip(borders, borders[1:])}

    def append(self, x, y):
        """ add a value to the Binner

        :param value: the value is appended to the appropriate bin
        """
        for (minimum, maximum) in self._data.keys():
            if x >= minimum and x < maximum:
                self._data[(minimum, maximum)].append(y)
                break

    def map(self, func):
        """ apply a function to the arrays inside each bin

        :param func: takes a list a argument and returns something.
        :return: a list of the bin's data applied to func, ordered by keys.
        """
        return [func(self._data[key])
                for key in sorted(self._data.keys())]

    def domain(self):
        """
        :return: the centers of the intervals (keys of _data)
        """
        return [0.5 * (minimum + maximum)
                for (minimum, maximum) in sorted(self._data.keys())]

    def merge(self):
        """ merge all data in _data

        :return: a list
        """
        return itertools.chain(self._data.values())

    @staticmethod
    def binary_map_binner(binner_1, binner_2, callback):
        """ apply a function of two arguments on each bin of two binner
        instances

        :param binner_1: instance of Binner
        :param binner_2: instance of Binner with the same intervals as binner_1
        :param callback: function that takes to lists, and anything
        :return: list of anything
        """
        return [callback(binner_1._data[key], binner_2._data[key])
                for key in sorted(binner_1._data.keys())]


def extent(x):
    """
    :return: first and last element of indexable sequence as a tuple
    """
    return (x[0], x[-1])


def fit_one_parameter(curve_a, curve_b):
    """ the second curve is scaled such that the difference of the two curves
    is minimized.

    :param curve_a: one dimensional arrays
    :param curve_b: one dimensional arrays with the same number of elements as
        curve_a
    :return: fit parameter
    """
    assert len(curve_a) == len(curve_b)
    data_matrix = np.concatenate([[curve_b], [np.zeros_like(curve_b)]], axis=0)
    solution_hint = np.linalg.lstsq(data_matrix.transpose(), curve_a)[0]
    assert solution_hint[1] < 0.00001
    return solution_hint[0]


def curve_fit(template, x, y):
    """ convenient helper function for fitting a curve, see scipy.curve_fit

    :param template: a function taking 1 + n arguments, with n being the number
        of free parameters
    :param x: a one dimensional array
    :param y: a one dimensional array with the same shape as x
    :return: triple: (estimated parameters, estimated errors, fitted function)
    """
    from scipy.optimize import curve_fit
    (params, errors) = curve_fit(template, x, y)
    try:
        errors = np.sqrt(np.diag(errors))
    except:
        pass
    return (params, errors, lambda t: template(t, *params))



def linear_fit(xx, yy):
    """ same as :func:`misc.curve_fit`,

    >>> misc.curve_fit(lambda a * x + b, xx, yy)
    """
    def template(x, a, b):
        return a * x + b
    return curve_fit(template, xx, yy)


def linear_fit_offset(offset, x, y):
    """ same as :func:`misc.curve_fit`,

    >>> misc.curve_fit(lambda a * x + offset, xx, yy)
    """
    def template(xx, a):
        return a * xx + offset
    return curve_fit(template, x, y)


class Log:
    """
    simple logging facility. Call for adding data with description, then print
    it

    :param title: a description string for the log book
    """

    def __init__(self, title):
        self.title = title
        self.info = collections.OrderedDict({})

    def __call__(self, description, data):
        """ add data with description

        :param description: string
        :param data: arbitrary data
        """
        self.info[description] = data

    def __str__(self):
        """ print a representation of the logged data """
        res = self.title
        res += '\n' + ''.join(['-' for i in range(len(self.title))]) + '\n'
        return res + "\n".join(["{0}:\t{1}".format(*item)
            for item in self.info.items()])




def phase_jumps(phases):
    """ determine phase jumps.

    :param phases: array of numbers, be monotonically increasing or decreasing
    :return: indices where phi is about zero.
    """
    indices = None
    if phases[-1] > phases[0]:
        indices = np.where(np.diff(phases % (2 * np.pi)) < 0)[0]
    else:
        indices = np.where(np.diff(phases % (2 * np.pi)) > 0)[0]
    return indices


def wrap(phase):
    """ wrap something into an interval

    :param phase: one dimensional array
    """
    import numpy as np
    return (np.array(phase) + np.pi) % (2 * np.pi ) - np.pi




def unzip(lll):
    """
    :param lll: is a list of lists. return a tuple of lists.
        equivalent to transpose in matrix language.
    """
    res = []
    initialized = False
    for ll in lll:
        for (i, l) in enumerate(ll):
            if not initialized:
                res.append([])
            res[i].append(l)
        initialized = True
    return res


class Toggler(object):
    """ a list of values is run through in a cyclic manner

    :param items: a list of arbitrary values
    """
    def __init__(self, items):
        self._items = items
        self._num = len(items)
        self._which = -1

    def which(self):
        """ get a value """
        self._which += 1
        return self._items[self._which % self._num]


def filter2dict(values, treshold=lambda v: v >= 1):
    """
    :param values: whatever iterable.
    :param treshold: a function, returning bool given an element of values.
    :return: what is not filtered out as a dictionary with original indices as
        keys and values as values.
    """
    return {i: v for (i, v) in enumerate(values) if treshold(v)}


def absolute_filter(values, treshold=0.5):
    """
    :param values: real valued numbers, a numpy array.
    :param treshold: take only components with a coefficient that is bigger than that
    :return: what is not filtered out as a dictionary with original indices as
        keys and values as values.
    """
    return filter2dict(values, treshold=lambda v: abs(v) >= treshold)


def relative_filter(values, treshold=0.5):
    """
    :param values: real valued numbers, a numpy array.
    :param treshold: take only components with a coefficient that is bigger than the
        with treshold scaled maximum.
    :return: what is not filtered out as a dictionary with original indices as
        keys and values as values.
    """
    import numpy as np
    maximal_component = np.amax(np.abs(values))
    return filter2dict(values, lambda v: abs(v) > treshold * maximal_component)

# def linearDependent(r1, r2, r3, r4):
#     from numpy import array
#     from numpy.linalg import det
#     m = array([r1-r2, r1-r3, r1-r4])
#     return det(m) < 0.00000001
# 
# 
# def cycleArray(arr, n):
#     """
#     takes elements 1 to the nth and put them to the tail.
#     """
#     from numpy import array
#     res = []
#     for i in [i + n for i in range(len(arr))]:
#         res.append(arr[i % len(arr)])
#     return array(res)
# 
# 
# def unwrap(arr, discont=3.13):
#     from numpy import zeros
#     res = zeros((len(arr)))
#     currentUnwrap = 0
#     for j in range(len(arr) - 1):
#         res[j] = arr[j] + currentUnwrap
#         dist = arr[j + 1] - arr[j]
#         if dist > discont:
#             currentUnwrap -= dist
#         elif dist < -discont:
#             currentUnwrap -= dist
#     res[-1] = arr[-1] + currentUnwrap
#     return res



def takeOneOf(n, ll):
    """ stroboscopic projection of array

    :param n: return a list with n elements
    :param ll: every n'th element of this list is returned
    :return: a list of every nth element of ll
    """
    i = len(ll)
    res = []
    for l in ll:
        if i % n == 0:
            res.append(l)
        i += 1
    return res


def expVals(init=1,vals=10,factor=1.1):
    """ return ``vals`` exponentially spaced values from ``init`` with the
    exponent ``factor`` """
    def expValsH(n,val):
        if n == 1:
            return [val*factor]
        else:
            return [val*factor]+expValsH(n-1,val*factor)
    return expValsH(vals,init/factor)


def rangedExpVals(low=1, high=500, n=5):
    """ return ``n`` exponentially spaced values from ``low`` to ``high`` """
    from numpy import exp, log
    factor = exp((log(high) - log(low)) / (n - 1))
    return expVals(init=low, vals=n, factor=factor)


