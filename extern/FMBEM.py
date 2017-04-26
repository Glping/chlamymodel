import fio
from FMM.inputDat import writeInputAndRemembery_NEW
from FMM.inputDat import writeInputAndRemembery_centered
import FMM.readOutput as ro
import FMM.meshObject as mo
import FMM.extractquantities as extr

from multiprocessing import Pool
import numpy as np
import os
import os.path
import shutil


def runit(system, objectnames, foldername, description,
          centered=False, objectname=None):
    """
    run simulation and obtain forces and torques of objectnames.
    """

    if centered:
        if objectname is None:
            raise Exception('identifier must be given when centering a system')
        data = runFMBEM_centered(foldername, system, objectname,
                                 objectnames, description=description)
    else:
        data = runFMBEM(foldername, system, objectnames,
                              description=description)

    forces = [extr.force(data[o].forces)
                    for o in objectnames]
    torques = [extr.centertorque(data[o].coordinates, data[o].forces)
                    for o in objectnames]

    if objectnames == []:
        return []
    return np.concatenate(forces + torques)


def run_sequential(recipes, objectnames, processes=8, folder='.'):
    """
    recipes is a dictionary. The keys represent identifiers of the systems,
    that are provided by the values. The keys are used as foldernames.
    The recipes are converted to meshes, and the respective simulation are run.
    The result are read and forces and torques are calculated, and saved in a
    dictionary containing the same keys as recipes.
    """
    jobs = {}

    for (k, recipe) in recipes.items():

        jobs[k] = runit(recipe, objectnames,
                       '{0}/{1}'.format(folder, k),
                       '{0}/{1}'.format(folder, k))

    return jobs


def run_parallel(recipes, objectnames, processes=8, folder='.'):
    """
    recipes is a dictionary. The keys represent identifiers of the systems,
    that are provided by the values. The keys are used as foldernames.
    The recipes are converted to meshes, and the respective simulation are run.
    The result are read and forces and torques are calculated, and saved in a
    dictionary containing the same keys as recipes.
    """
    jobs = {}

    with Pool(processes=processes) as pool:

        for (k, recipe) in recipes.items():

            jobs[k] = pool.apply_async(runit,
                          [recipe,
                           objectnames,
                           '{0}/{1}'.format(folder, k),
                           '{0}/{1}'.format(folder, k)])

        # run them all
        res = {k: job.get() for (k, job) in jobs.items()}

    return res


def runFMBEM(folder, system, objects, description='description'):
    """
    run the FMBEM simulation given by system, which is a dictionary with a
    key, that is provided by FMM.meshObject.transformedShape. For details
    on the creation of the data structure. Input data is created in folder,
    folder is created if necessary. An input.cnd is created.
    This function also analyses the output of the simulation and returns the
    forces and velocities, coordinates and areas (distributions) of the
    specified objects.
    objects is a list of strings, representing coordinate and triangulation
    ranges, saved in remembery.py
    """
    # create data files
    oldfolder = os.getcwd()
    fio.makedir(folder)
    os.chdir(folder)

    with open('remembery.py'.format(folder), 'w') as remembery:
        writeInputAndRemembery_NEW(system,
                                   remembery=remembery,
                                   description=description)

    shutil.copyfile(os.path.expanduser('~/.FMBEM/rough.cnd'), 'input.cnd')

    FMBEM('.')

    res = { o: ro.extractDataByName(o) for o in objects }

    os.chdir(oldfolder)

    return res


def runFMBEM_centered(folder, system, objectname, objects, description='description'):
    """
    like run_FMBEM but the whole mesh is moved such that the origin of the
    system is also the origin of the submesh representing objectname.
    """
    # create data files
    oldfolder = os.getcwd()
    fio.makedir(folder)
    os.chdir(folder)

    remembery = open('remembery.py'.format(folder), 'w')
    writeInputAndRemembery_centered(system,
                               objectname,
                               remembery=remembery,
                               description=description)
    remembery.close()

    shutil.copyfile(os.path.expanduser('~/.FMBEM/rough.cnd'), 'input.cnd')

    FMBEM('.')

    res = { o: ro.extractDataByName(o) for o in objects }

    os.chdir(oldfolder)

    return res


def FMBEM(folder):

    def linenumbers(filename):
        try:
            filehandle = open(filename, 'r')
            number = len(filehandle.readlines())
            filehandle.close()
            return number
        except:
            return 0

    def call_until_success():

        errorhandle = open('FMBEMerrors', 'a')
        devnull = open('/dev/null', 'w')

        success = subprocess.call(
                #['wine', os.path.expanduser('~/.FMBEM/3D_Stokes_Flow_FMM.exe')],
                [os.path.expanduser('~/.FMBEM/3D_Stokes_Flow_FMM_linux.exe')],
                stdout=devnull, stderr=errorhandle)
        if success == 1:
            errorhandle.write('error executing wine')
        devnull.close()

        if linenumbers('output.dat') < 10:
            errorhandle.write('program did not finish, restarting...')
            errorhandle.close()
            return call_until_success()
        errorhandle.close()

    import subprocess
    import time
    oldfolder = os.getcwd()
    os.chdir(folder)

    call_until_success()
    os.chdir(oldfolder)


if __name__ == '__main__':

    positions1 = [(i, np.sin(0.1 * i), 0) for i in range(10)]
    positions2 = [(i, np.sin(0.1 * (i + 1)), 0) for i in range(10)]
    flagellum = {'type': 'flagellum',
                 'positions': positions1,
                 'future positions': positions2,
                 'radius': 1,
                 'dt': 1,
                 'azimuth grid': 5}
    res = runFMBEM('flagellumtest',
                mo.transformedShape({'flagellum':
                  mo.transformedShape(flagellum)}),
             ['flagellum'], description='a flagellum')
    print(res['flagellum'].coordinates)
