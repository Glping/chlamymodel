"""
single swimmer simulations, parallelized. I am not specific about the type of
swimmer used.
"""

import multiprocessing
import itertools
import shutil
from copy import deepcopy
import numpy as np

import extern.FMBEM as FMBEM
import FMM.handlerecipes as handlerecipes


def friction_factor():
    """
    the friction matrix is obtained by rigid body movements. In the case of
    rotations, the unit rotation 1, would lead to a strange velocity
    distribution, of some strangely deforming body. Use the friction factor
    to decrease this deformation. Also the friction matrix has to be rescaled.
    """
    return 10000


def orthogonal_movement_recipes(creature, objectnames):
    """
    create timestep recipes, given the shape changing recipe.
    The creature may have a velocity or angular velocity.
    The movement unit is 1 / 10000 (friction_factor()).
    =objectnames= refer to the submeshes, representing freely swimming
    creatures.
    Return a dict, with keys being the potential foldernames and values being
    the recipes. The keys are also used to decide, if a system has to be
    centered, before it is run.
    This function handles an arbitrary number of particles.
    """

    rigid_creature = handlerecipes.rigify_flagella(creature)

    res = {}
    for (abbrev, state) in zip(['v', 'w'], ['velocity', 'angular']):
        for objectname in objectnames:
            for (i, direction) in enumerate(['x', 'y', 'z']):

                amount = [0, 0, 0]
                amount[i] = 1 / friction_factor()
                index = '{0}{1}_{2}'.format(abbrev, direction, objectname)
                #index = abbrev + str(pid + 1) + direction
                res[index] = handlerecipes.modify_recipe(
                        rigid_creature,
                        objectname,
                        lambda sys: handlerecipes.increase_state(sys,
                                                                 state,
                                                                 amount),
                        outside=False)
    return res


def friction_from_results(results, objectnames):
    """
    the results from the timestep calculations are saved in a dictionary. It is
    intepreted, and the friction matrix is returned.
    """
    folders = ['{0}{1}_{2}'.format(tp[0], tp[2], tp[1])
               for tp in itertools.product(
                        ['v', 'w'],
                        objectnames,
                        ['x', 'y', 'z'])]

    particle_number = len(objectnames)

    friction = np.zeros((6 * particle_number, 6 * particle_number))
    for (i, fold) in enumerate(folders):
        friction[:, i] = results[fold]

    return friction_factor() * friction


def run_timestep(timestep, creature, objectnames,
                 processes=8, removefolder=True):
    """
    timestep: a number, used for creating the foldername.
    creature: a usable system recipe.
    objectnames: list of submesh identifiers, representing free particles.
    processes: number of processes to be used for the calculation.
    removefolder: if it is True, remove the created folders.

    create recipes, run them all in appropriate folders, obtain force, torque
    and friction matrix, remove created folders.
    force, torque and friction are returned.
    """
    def the_sequential(rcps):
        """
        does the same as the_parallel but not in parallel. It is for testing
        purposes.
        """
        jobs = {}
        for (key, recipe) in rcps.items():

            # if key[:2] in [h[0] + h[1] for h
            #         in itertools.product(['v', 'w'], ['x', 'y', 'z'])]:
            #     jobs[key] = FMBEM.runit(
            #             recipe,
            #             objectnames,
            #             '{0}/{1}'.format(foldername, key),
            #             '{0}/{1}'.format(foldername, key),
            #             centered=True,
            #             objectname=key[3:])
            # else:
                jobs[key] = FMBEM.runit(
                        recipe,
                        objectnames,
                        '{0}/{1}'.format(foldername, key),
                        '{0}/{1}'.format(foldername, key))

        return jobs


    def the_parallel(rcps):
        """
        execute the recipes given as argument. rcps is a dictionary. Depending
        on the key string, the simulation is executed centered at some object
        or not.
        """
        jobs = {}
        with multiprocessing.Pool(processes=processes) as pool:

            for (key, recipe) in rcps.items():

                # if key[:2] in [h[0] + h[1] for h
                #     print('if')
                #         in itertools.product(['v', 'w'], ['x', 'y', 'z'])]:
                #     jobs[key] = pool.apply_async(FMBEM.runit,
                #             [recipe,
                #              objectnames,
                #              '{0}/{1}'.format(foldername, key),
                #              '{0}/{1}'.format(foldername, key)],
                #             {'centered': True,
                #              'objectname': key[3:]})
                # else:
                    jobs[key] = pool.apply_async(FMBEM.runit,
                            [recipe,
                             objectnames,
                             '{0}/{1}'.format(foldername, key),
                             '{0}/{1}'.format(foldername, key)])

            return {key: job.get() for (key, job) in jobs.items()}


    # the keys of the dictionary that is returned by the following function
    # are: [vw][xyz]_objectname
    recipes = orthogonal_movement_recipes(creature, objectnames)
    recipes['time'] = creature

    foldername = 'step{0:04d}'.format(timestep)

    if processes > 1:
        forces = the_parallel(recipes)
    else:
        forces = the_sequential(recipes)

    friction = friction_from_results(forces, objectnames)

    if removefolder:
        shutil.rmtree(foldername)

    return (forces['time'], friction)
