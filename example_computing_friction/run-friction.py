"""
run simulation with one Chlamy at a certain phase of its left and right
flagellum. Perform movements in a certain generalized velocity and calculate
the corresponding generalized forces.

Collecting the results of simulations of all the different velocities leads
to the construction of the friction matrix.
"""

import sys
import numpy as np
import FMM.chlamyinput as ci
import FMM.handlerecipes as hr
import FMM.extractquantities as xt
from extern.FMBEM import runFMBEM



def get_fields(phase_left, phase_right,
               amplitude_left, amplitude_right, dof):
    """
    return a column of the generalized friction matrix
    """

    frequency = 50
    period = 1 / frequency
    omega = 2 * np.pi * frequency
    dA = 0.001
    time_resolution = 100
    spatial_resolution = 20
    azimuth_grid = 6
    body_grid = 20

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

    amplitude_left = min(amplitude_left, amp_with_phi(phase_left))
    amplitude_right = min(amplitude_right, amp_with_phi(phase_right))

    phase_left_2 = phase_left
    phase_right_2 = phase_right
    amplitude_left_2 = amplitude_left
    amplitude_right_2 = amplitude_right
    velocity = (0, 0, 0)
    angular = (0, 0, 0)

    if dof == 'phase_left':
        phase_left_2 = phase_left + 2 * np.pi / time_resolution
        q = omega
    elif dof == 'phase_right':
        phase_right_2 = phase_right + 2 * np.pi / time_resolution
        q = omega
    elif dof == 'amplitude_left':
        amplitude_left_2 = amplitude_left + dA
        q = dA / (period / time_resolution)
    elif dof == 'amplitude_right':
        amplitude_right_2 = amplitude_right + dA
        q = dA / (period / time_resolution)
    elif dof == 'x':
        velocity = (1, 0, 0)
        q = 1
    elif dof == 'y':
        velocity = (0, 1, 0)
        q = 1
    elif dof == 'orientation':
        angular = (0, 0, 0.01)
        q = 0.01
    else:
        print('third argument is wrong!', file=sys.stderr)


    data_l_p1_a1 = ci.ruloff_chlamy(
        '20150602_092953_01',
        phase_left, amplitude_left,
        spatial_resolution=spatial_resolution)

    data_l_p2_a2 = ci.ruloff_chlamy(
        '20150602_092953_01',
        phase_left_2, amplitude_left_2,
        spatial_resolution=spatial_resolution)

    data_r_p1_a1 = ci.ruloff_chlamy(
        '20150602_092953_01',
        phase_right, amplitude_right,
        spatial_resolution=spatial_resolution)

    data_r_p2_a2 = ci.ruloff_chlamy(
        '20150602_092953_01',
        phase_right_2, amplitude_right_2,
        spatial_resolution=spatial_resolution)

    recipe_flagellum_l = hr.transformed_shape({
        'type': 'flagellum',
        'radius': 0.2,#data_l_p1_a1['length'] / 50,
        'positions':
            [(x, y, 0) for (x, y) in zip(
                data_l_p1_a1['xflagL'], data_l_p1_a1['yflagL'])],
        'future positions':
            [(x, y, 0) for (x, y) in zip(
                data_l_p2_a2['xflagL'], data_l_p2_a2['yflagL'])],
        'azimuth grid': azimuth_grid,
        'dt': period / time_resolution})

    recipe_flagellum_r = hr.transformed_shape({
        'type': 'flagellum',
        'radius': 0.2,#data_r_p1_a1['length'] / 50,
        'positions':
            [(x, y, 0) for (x, y) in zip(
                data_r_p1_a1['xflagR'], data_r_p1_a1['yflagR'])],
        'future positions':
            [(x, y, 0) for (x, y) in zip(
                data_r_p2_a2['xflagR'], data_r_p2_a2['yflagR'])],
        'azimuth grid': azimuth_grid,
        'dt': period / time_resolution})

    recipe_body = hr.transformed_shape({
        'type': 'ellipsoid',
        'lengths': (data_l_p1_a1['body'][0],
                    data_l_p1_a1['body'][1],
                    data_l_p1_a1['body'][0]),
        'axe1': (1, 0, 0),
        'axe2': (0, 1, 0),
        'grid': body_grid})


    chlamy = hr.transformed_shape({
        'left': recipe_flagellum_l,
        'right': recipe_flagellum_r,
        'body': recipe_body}, velocity=velocity, angular=angular)


    fields = runFMBEM(
        '.', chlamy, ['all'],
        description='left: ({0:2.2f}, {2:2.2f}), right: ({1:2.2f}, {3:2.2f})'.format(
            phase_left, phase_right,
            amplitude_left, amplitude_right))['all']

    return (fields.velocities / q, fields.forces / q)


PHASE_LEFT = float(sys.argv[1])
PHASE_RIGHT = float(sys.argv[2])
AMPLITUDE_LEFT = float(sys.argv[3])
AMPLITUDE_RIGHT = float(sys.argv[4])

DOF = sys.argv[5]
get_fields(PHASE_LEFT, PHASE_RIGHT, AMPLITUDE_LEFT, AMPLITUDE_RIGHT, DOF)
sys.exit()

FIELDS = [get_fields(PHASE_LEFT, PHASE_RIGHT,
                     AMPLITUDE_LEFT, AMPLITUDE_RIGHT,
                     dof)
          for dof in ['phase_left', 'phase_right',
                      'amplitude_left', 'amplitude_right',
                      'x', 'y', 'orientation']]

VELOCITIES = [f[0] for f in FIELDS]
FORCES = [f[1] for f in FIELDS]

FRICTION_MATRIX = [[xt.dissipation(v, f)
                    for f in FORCES]
                   for v in VELOCITIES]

np.savetxt('friction', FRICTION_MATRIX)
