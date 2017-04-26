"""
modify mesh building recipes.
"""

from copy import deepcopy
import numpy as np


def transformed_shape(system,
                     translation=(0, 0, 0),
                     rotation=np.diag([1, 1, 1]),
                     velocity=(0, 0, 0),
                     angular=(0, 0, 0)):
    """
    if system is a dict with keys being valid recipes, return a valid recipe.
    """
    return {'system': system,
            'rotate': rotation,
            'translate': translation,
            'velocity': velocity,
            'angular': angular}


def modify_recipe(system, objectname, modifier, outside=False):
    """
    recursively walk through a recipe. modify all subsystems that are matched
    by the objectname by modifier.
    =system= is a valid recipe.
    If outside is true, properties like velocity, angular, translation and
    rotation can be changed. Type specific properties are changed with
    outdie=True
    """
    res = deepcopy(system)

    if 'type' in system['system']:
        return res

    for rid in system['system']:
        if rid == objectname:
            if outside:
                res = modifier(system)
            else:
                res['system'][rid] = modifier(system['system'][rid])
        else:
            res['system'][rid] = modify_recipe(system['system'][rid],
                                               objectname, modifier)

    return res


def type_query(thetype):
    """
    return the query for extracting object names of a certain type
    (e.g. flagellum)
    """
    def query(sys):
        """ this is what is to be returned """
        if 'type' in sys['system']:
            return sys['system']['type'] == thetype
        return False
    return query


def query_recipe(system, query):
    """
    query is a function that return a bool. If it does, return the
    object identifier name.
    a list of such names is returned.
    """
    res = []
    for rid in system['system']:
        if type(system['system'][rid]) != dict:
            continue
        if query(system['system'][rid]):
            res.append(rid)
        else:
            res = res + query_recipe(system['system'][rid], query)
    return res


def query_flagella(system):
    """
    return all object identifier names of occuring flagella in the mesh recipe.
    """
    return query_recipe(system, type_query('flagellum'))


def set_state(sys, state, value):
    """ set a state variable of the system given """
    return modify_state(sys, state, lambda val: value)


def modify_state(sys, state, func):
    """ modify the state of a sys by applying func """
    sysh = deepcopy(sys)
    sysh[state] = func(sys[state])
    return sysh


def increase_state(sys, state, amount):
    """ it is a typical scenario to increase the velocity """
    return modify_state(sys, state, lambda val: val + np.array(amount))


def set_movement(sys, velocity, angular):
    """ set (angular) velocity of a system """
    return set_state(set_state(sys, 'velocity', velocity), 'angular', angular)


def set_sub_movement(system, objectname, velocity, angular):
    """ set the (angular) velocity of =objectname= """
    return modify_recipe(system, objectname,
                         lambda sys: set_movement(sys, velocity, angular))


def rigify_flagella(system):
    """
    set the =future positions= equal to the =positions=. If this function is
    called with a system, whose object =flagellumname= is not a recipe of type
    flagellum, there is a KeyError.
    """

    def setter(sys):
        """ status quo for the flagellar movement """
        sys['system']['future positions'] = sys['system']['positions']
        return sys

    system_h = deepcopy(system)
    for name in query_flagella(system):
        system_h = modify_recipe(system_h, name, setter)

    return system_h


if __name__ == '__main__':

    # create a simple recipe
    ellipsoid = {'type': 'ellipsoid',
                 'position': (0, 0, 0),
                 'lengths': (2, 2, 4),
                 'axe1': (1, 0, 0),
                 'axe2': (0, 1, 0),
                 'grid': 20}
    xx = np.linspace(0, 30, 10)
    positionsL1 = [(x, y, z) for (x, y, z) in zip(np.sin(xx), np.cos(xx), xx)]
    positionsL2 = [(x, y, z) for (x, y, z) in zip(np.sin(xx + 0.1), np.cos(xx), xx)]
    flagellaL = {'type': 'flagellum',
                 'positions': positionsL1,
                 'future positions': positionsL2,
                 'radius': 0.4,
                 'dt': 1,
                 'azimuth grid': 6}
    SYSTEM = {'head': transformed_shape(ellipsoid, translation=(0, 0, -4.5)),
              'tail': transformed_shape(flagellaL)}
    SYSTEM = transformed_shape(SYSTEM)



    # check rigify_flagellum
    def prnt(sys):
        print(sys['system']['future positions'] == sys['system']['positions'])

    modify_recipe(SYSTEM, 'tail', prnt)
    rigidsystem = rigify_flagella(SYSTEM)
    modify_recipe(rigidsystem, 'tail', prnt)
