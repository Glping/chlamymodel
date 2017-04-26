from copy import deepcopy
import numpy as np
import itertools


class Binner:

    def __init__(self, bins, range_):
        """
        Keyword Arguments:
        bins -- number of bins
        rng  -- range, tuple of minimum and maximum
        """
        borders = np.linspace(range_[0], range_[1], num=bins+1)
        self._data = {b: [] for b in zip(borders, borders[1:])}

    def append(self, x, y):
        """
        Keyword Arguments:
        value -- scalar value to be appended in the appropriate bin
        """
        for (minimum, maximum) in self._data.keys():
            if x >= minimum and x < maximum:
                self._data[(minimum, maximum)].append(y)
                break

    def map(self, func):
        """
        Keyword Arguments:
        func -- takes a list a argument and returns something else
        return a list of the bins data applied to func, ordered by keys.
        """
        return [func(self._data[key])
                for key in sorted(self._data.keys())]

    def domain(self):
        """
        return the centers of the intervals (keys of _data)
        """
        return [0.5 * (minimum + maximum)
                for (minimum, maximum) in sorted(self._data.keys())]

    def merge(self):
        """
        merge all data in _data and return a list
        """
        return itertools.chain(self._data.values())

    def binary_map_binner(binner_1, binner_2, callback):
        """
        Keyword Arguments:
        binner_1 -- instance of Binner
        binner_2 -- instance of Binner with the same intervals as binner_1
        callback -- function that takes to lists, and returns of
        does something with it.
        """
        return [callback(binner_1._data[key], binner_2._data[key])
                for key in sorted(binner_1._data.keys())]


if __name__ == '__main__':

    import quick

    binner = Binner(5, (0, 2 * np.pi))
    for v in [0, 1, 1, 2, 3, 2, 1]:
        binner.append(v, 777)

    quick.plot(binner.domain(), binner.map(np.sum))


def extent(x):
    """ return first and last element of indexable sequence as a tuple """
    return (x[0], x[-1])


def fit_one_parameter(curve_a, curve_b):
    """
    curve_a and curve_b are one dimensional arrays with the same number of
    elements.
    """
    assert len(curve_a) == len(curve_b)
    data_matrix = np.concatenate([[curve_b], [np.zeros_like(curve_b)]], axis=0)
    solution_hint = np.linalg.lstsq(data_matrix.transpose(), curve_a)[0]
    assert solution_hint[1] < 0.00001
    return solution_hint[0]


def curve_fit(template, x, y):
    from scipy.optimize import curve_fit
    (params, errors) = curve_fit(template, x, y)
    try:
        errors = np.sqrt(np.diag(errors))
    except:
        pass
    return (params, errors, lambda t: template(t, *params))



def linear_fit(xx, yy):
    def template(x, a, b):
        return a * x + b
    return curve_fit(template, xx, yy)


def linear_fit_offset(offset, x, y):
    def template(xx, a):
        return a * xx + offset
    return curve_fit(template, x, y)


class Log:

    def __init__(self, title):
        self.title = title
        self.info = collections.OrderedDict({})

    def __call__(self, description, data):
        self.info[description] = data

    def __str__(self):
        res = self.title
        res += '\n' + ''.join(['-' for i in range(len(self.title))]) + '\n'
        return res + "\n".join(["{0}:\t{1}".format(*item)
            for item in self.info.items()])




def find_good_borders(bad_indices):
    """
    having a list of integers, for each element, find the surrounding integers
    that are not in the list.
    """
    bad_indices = set(bad_indices)

    def count_func(func):
        def ret_func(index):
            counter = 1
            while True:
                if func(index, counter) not in bad_indices:
                    return func(index, counter)
                counter += 1
        return ret_func

    count_down = count_func(lambda a, b: a - b)
    count_up = count_func(lambda a, b: a + b)

    res = []
    for (i, bi) in enumerate(sorted(bad_indices)):
        g1 = count_down(bi)
        g2 = count_up(bi)
        res.append((bi, (g1, g2)))

    return res


def bad_from_good_indices(good_indices):
    """
    given a list of integers, return all integers from holes in good_indices
    """
    counter = good_indices[0]
    res = []
    leave_flag = False
    for i in good_indices:
        while True:
            if i == counter:
                counter += 1
                leave_flag = True
                break
            else:
                res.append(counter)
                counter += 1
        if leave_flag:
            leave_flag = False
            continue
    return res

def good_from_bad_indices(bad_indices, last_index):
    """
    given a list of integers, return a list starting from zero, not containing
    the elements of bad_indices.
    """
    bad_indices = set(bad_indices)
    return [i for i in range(last_index + 1) if i not in bad_indices]


def ats(ii, ll):
    """
    list elements ii of ll are returned
    """
    return [ll[i] for i in ii]


def phase_jumps(phases):
    """
    be phases some phase dynamics. It may be monotonically increasing or
    decreasing.
    Return indices where phi is about zero.
    """
    indices = None
    if phases[-1] > phases[0]:
        indices = np.where(np.diff(phases % (2 * np.pi)) < 0)[0]
    else:
        indices = np.where(np.diff(phases % (2 * np.pi)) > 0)[0]
    return indices


def remove_list_elements(lists, indexer):
    """
    lists is a list of lists.
    indexer is a function that returns a list of indices, given lists[0].
    remove all the elements of the elements of lists at positions given by
    indexers indices.
    this function is not state changing, but returns the modified list of lists.
    """
    bad_indices = indexer(lists[0])
    return [remove_indices(bad_indices, l) for l in lists]


class Struct(object):

    def __init__(self, **entries): 
        self.__dict__.update(entries)


def split_interval(number, ll):
    """
    split a list ll into a list of lists, each with number elements. elements
    of sublists are dropped if there are not sufficient elements.
    """
    return [ll[ i * number : (i + 1) * number ]
            for i in range(int(len(ll) / number))]


def pickled_calculation(calculation, filename, force=False):
    """
    perform a calculation, if the file filename does not exist , and pickle the
    result. If the file filename exists, load the pickle file.
    if force is True, do the calculation also if the pickle file exists.
    """
    import pickle
    data = None
    try:
        with open(filename, 'rb') as filehandle:
            data = pickle.load(filehandle)
    except:
        data = calculation()
        with open(filename, 'wb') as filehandle:
            pickle.dump(data, filehandle)

    return data


def find_holes(sequence):
    """
    sequence: of integers
    find holes.
    e.g.:
    [1, 2, 3, 6, 7, 9]
    -> [2, 4]
    """
    intervals = []
    current = sequence[0]
    current_index = 0
    for (i, s) in enumerate(sequence[1:]):
        if s == current + 1:
            current = s
            current_index = i + 1
            continue
        else:
            intervals.append(current_index)
            current = s
            current_index = i + 1
    return intervals




def debug(a, f=lambda a: a):
    """
    print a representation given by f of an arbitrary value and return it.
    """
    print(f(a))
    return a


def debug_shape(a):
    """
    print the shape of an array and return the array.
    """
    return debug(a, lambda a: a.shape)


def wrap(phase):
    """ wrap something into an interval """
    import numpy as np
    return (np.array(phase) + np.pi) % (2 * np.pi ) - np.pi


def remove_indices(indices, ll):
    """ return a new list with the element at positions indices removed """
    res = []
    indexset = set(indices)
    for (i, l) in enumerate(ll):
        if i not in indexset:
            res.append(l)
    return res


def split_indices(ll, condition):
    """
    return a tuple of indices of the list ll, of elemtns, fulfilling and
    not fulfilling condition.
    """
    i_1 = []
    i_2 = []
    for (i, l) in enumerate(ll):
        if condition(l):
            i_1.append(i)
        else:
            i_2.append(i)
    return (i_1, i_2)


def at_custom_index(custom_index, custom_indices, ll):
    """ return the element of ll at index custom_index in custom_indices """
    indices = (i for (i, ci) in enumerate(custom_indices) if ci == custom_index)
    index = next(indices)
    return ll[index]


def join_lists(kk, k_indices, ll, l_indices):
    """
    join the lists kk and ll. The elements of kk are at positions indices in the
    resulting list.
    """
    #assert len(kk) == len(k_indices)
    #assert len(ll) == len(l_indices)

    res = []
    k_index = 0
    l_index = 0
    if len(k_indices) == 0:
        return ll
    if len(l_indices) == 0:
        return kk
    min_index = min(k_indices[0], l_indices[0])
    max_index = max(k_indices[-1], l_indices[-1])
    k_indexset = set(k_indices)
    l_indexset = set(l_indices)

    for i in range(min_index,
                   min_index + max_index):
        if i in l_indices:
            res.append(ll[l_index])
            l_index += 1
        elif i in k_indices:
            res.append(kk[k_index])
            k_index += 1
        else:
            assert (i not in k_indexset) and (i not in l_indexset)

    return res


def split_list(ll, condition):
    """
    return a tuple of indices of the list ll, of elemtns, fulfilling and
    not fulfilling condition.
    """
    l_1 = []
    l_2 = []
    for l in ll:
        if condition(l):
            l_1.append(l)
        else:
            l_2.append(l)
    return (l_1, l_2)



def find_indices(ll, condition):
    """
    ll: list
    condition: function from list element to boolean.
    return a list of indices, fullfilling condition.
    """
    return [i for (i, l) in enumerate(ll) if condition(l)]


def filter_modify(ll, condition, f_1, f_2):
    """
    ll is a list.
    condition is a function from list element to boolean.
    f_1 and f_2 are function from list element.
    for each element, if condition is fulfilled, save f_1(list element),
    otherwise f_2(list element).
    """
    res = []
    for l in ll:
        if condition(l):
            res.append(f_1(l))
        else:
            res.append(f_2(l))
    return res


def thunk(f, *args, **kwargs):
    def func():
        return f(*args, **kwargs)
    return func


def intersperse(ll_1, ll_2):
    """ having two lists, intersperse them. """
    res = []
    for (l_1, l_2) in zip(ll_1, ll_2):
        res.append(l_1)
        res.append(l_2)
    return res


def to_ranges(ll):
    """ a list of borders is turned into a list of ranges """
    if len(ll) == 2:
        return [ll]
    return zip(ll[:-1], ll[1:])


def which_range(ll, val):
    """
    ll is a list of ranges.
    val is a value.
    return the index of the range, in which the value is.
    return None, if val is outside any given range.
    """
    for (i, (l_1, l_2)) in enumerate(ll):
        if val >= l_1 and val < l_2:
            return i
    return None


def group_list(limits, good_indices, ll):
    """
    limits: the limits at which elements of ll have to be split, but not at
    their indices, but at their respective good_indices.
    good_indices and ll must have the same number of arguments.
    """
    rng = list(to_ranges(limits))
    res = [[] for i in rng]
    for (g_i, l) in zip(good_indices, ll):
        i = which_range(rng, g_i)
        res[i].append(l)
    return res



def unzip(lll):
    """
    ll is a list of lists. return a tuple of lists.
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
    """
    class is initialized with a few items and provides a function for obtaining
    a value. At each call of which, an item of items is returned, in a cyclic
    manner.
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
    values: whatever iterable.
    treshold: a function, returning bool given an element of values.
    return what is not filtered out as a dictionary with original indices as
    keys and values as values.
    """
    return {i: v for (i, v) in enumerate(values) if treshold(v)}


def absolute_filter(values, treshold=0.5):
    """
    values: real valued numbers, a numpy array.
    treshold: take only components with a coefficient that is bigger than that
    return what is not filtered out as a dictionary with original indices as
    keys and values as values.
    """
    return filter2dict(values, treshold=lambda v: abs(v) >= treshold)


def relative_filter(values, treshold=0.5):
    """
    values: real valued numbers, a numpy array.
    treshold: take only components with a coefficient that is bigger than the
              with treshold scaled maximum.
    return what is not filtered out as a dictionary with original indices as
    keys and values as values.
    """
    import numpy as np
    maximal_component = np.amax(np.abs(values))
    return filter2dict(values, lambda v: abs(v) > treshold * maximal_component)

def linearDependent(r1, r2, r3, r4):
    from numpy import array
    from numpy.linalg import det
    m = array([r1-r2, r1-r3, r1-r4])
    return det(m) < 0.00000001


def cycleArray(arr, n):
    """
    takes elements 1 to the nth and put them to the tail.
    """
    from numpy import array
    res = []
    for i in [i + n for i in range(len(arr))]:
        res.append(arr[i % len(arr)])
    return array(res)


def unwrap(arr, discont=3.13):
    from numpy import zeros
    res = zeros((len(arr)))
    currentUnwrap = 0
    for j in range(len(arr) - 1):
        res[j] = arr[j] + currentUnwrap
        dist = arr[j + 1] - arr[j]
        if dist > discont:
            currentUnwrap -= dist
        elif dist < -discont:
            currentUnwrap -= dist
    res[-1] = arr[-1] + currentUnwrap
    return res


def flip(tpl):
    return (tpl[1], tpl[0])

def takeOneOf(n, ll):
    """ take every nth element of a list ll into the list """
    i = len(ll)
    res = []
    for l in ll:
        if i % n == 0:
            res.append(l)
        i += 1
    return res

def lessenList(ll,pos):
    """
    take out position pos of list ll.
    return a list with len = len(ll)-1.
    """
    return ll[:pos]+ll[pos+1:]

def transposeIndex(ll,i1,i2):
    return ll[:i1] + [ll[i2]] + ll[i1+1:i2] + [ll[i1]] + ll[i2+1:]

def permutations(t):
    def permutationsH(t):
        """warning: does not return a list, but a generator.
        direct assignment has funny consequences!"""
        lt = list(t)
        lnt = len(lt)
        if lnt == 1:
            yield lt
        st = set(t)
        for d in st:
            lt.remove(d)
            for perm in permutations(lt):
                yield [d]+perm
            lt.append(d)
    return list(permutationsH(t))

def noverm (n,m):
    from math import factorial as fac
    return (fac(n)/(2**m*fac(m)*fac(n-2*m)))

def trinomial (n1,n2,n3):
    """ computes the trinomial coefficient """
    from math import factorial as fac
    return (fac(n1+n2+n3))/(fac(n1)*fac(n2)*fac(n3))


def reduce (fn,ll):
    """reduce takes a function of two arguments and a list"""
    if len(ll) == 1:
        return ll[0]
    return fn(ll[0],reduce(fn,ll[1:]))


def oddFac(nu):
    if nu == -1:
        return 1
    else:
        return nu*oddFac (nu-2)



def kroneckerList (ll):
    """ test, if a list, interpreted as indizes of a kronecker symbols,
    would return 0 (False) or 1 (True) """
    res = True
    i1 = 0
    i2 = 1
    while res:
        res &= ll[i1] == ll[i2]
        i1 += 2
        i2 += 2
        if i1 >= len (ll):
            break
    return res


def listQuotient(ll,kk):
    """ return a list containing elements of ll that are not present in kk """
    hh = []
    for l in ll:
        if l not in kk:
            hh.append(l)
    return hh

def zipList (ll,kk,pos):
    """rr is the resulting list with values of kk at
    positions pos. The other elements are of ll."""
    if len (kk) != len (pos):
        print ('misc:zipList: bad usage: position list and second list')
        print ('must have the same length to be meangingful.')
    rr = []
    counter = 0
    poscounter = 0
    for l in ll:
        while True:
            if poscounter >= len (pos):
                break
            if counter == pos[poscounter]:
                rr.append (kk[poscounter])
                poscounter += 1
                counter += 1
            else:
                break
        rr.append (l)
        counter += 1
    while True:
        if poscounter < len (pos):
            rr.append (kk[poscounter])
            poscounter += 1
        else:
            break
    return rr

def zipListWithDict(ll,dd):
    res = []
    listCounter = 0
    resCounter = 0
    sortedKeys = sorted(dd.keys())
    for k in sortedKeys:
        while resCounter < k:
            if listCounter >= len(ll):
                break
            res.append( ll[listCounter] )
            listCounter += 1
            resCounter += 1
        res.append( dd[k] )
        resCounter += 1
    return res+ll[listCounter:]

def repeatingList (ll):
    """ test if a list contains only equal values """
    l = len (ll)
    h = [ll[0]]*l
    return h == ll

def odd(nu):
    """ test if a number is odd """
    return nu & 1 and True or False

def oddMinus(nu):
    if odd(nu):
        return (-1)
    else:
        return 1

def repeat(f,n,g):
    """ do something (f) n times. combine the results via g """
    res = []
    for i in range(n):
        res.append(f())
    return reduce(lambda x,y: g(x,y),res)

def expVals(init=1,vals=10,factor=1.1):
    def expValsH(n,val):
        if n == 1:
            return [val*factor]
        else:
            return [val*factor]+expValsH(n-1,val*factor)
    return expValsH(vals,init/factor)

def rangedExpVals(low=1, high=500, n=5):
    from numpy import exp, log
    factor = exp((log(high) - log(low)) / (n - 1))
    return expVals(init=low, vals=n, factor=factor)

def compand(ll):
    """
    takes an array of numbers and return an
    array containing number between zero and one
    with the same mutual ratios
    """
    from numpy import array
    mx = max(ll)
    mn = min(ll)
    ll = array(ll) - mn
    ll = ll / (mx - mn)
    return ll


def run_parallel(jobs, processes=8):
    """
    jobs is a list of functions, that take no arguments.
    Execute them in parallel with 'processes' processors.
    """
    from multiprocessing import Pool
    pool = Pool(processes=processes)
    pending_jobs = [pool.apply_async(j) for j in jobs]
    return [j.get() for j in pending_jobs]


class StateSwitcher(object):
    """
    a state is represented as a superposition of booleans. Each subspace can
    be on or off.
    """

    def __init__(self, number_of_states, current_state=None):
        if current_state is None:
            self.current_state = [False] * number_of_states
        else:
            self.current_state = current_state
        self.states = [deepcopy(self.current_state)]

    def switch(self, num):
        """ switch sub state num """
        self.current_state[num] = not self.current_state[num]
        self.states.append(deepcopy(self.current_state))

    def state_to_integer(self, state):
        """ convert single state """
        res = 0
        for (i, substate) in enumerate(state):
            if substate:
                res += 2 ** i
        return res

    def current_to_integer(self):
        return self.state_to_integer(self.current_state)

    def to_integer(self):
        """ return a list of integers, representing the states. """
        return map(self.state_to_integer, self.states)


