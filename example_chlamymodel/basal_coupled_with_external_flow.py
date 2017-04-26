"""
do simulations for finding the synchronization strength of an elastically
clamped cell with amplitude compliance.
Interesting values for the clamping elasticity are from 1 to 10^6
"""
import sys
import numpy as np

import chlamymodel.seven as dyn
import chlamymodel.basal as basal
import chlamymodel.drivers as drivers
import config


frequency = config.frequency
omega = config.omega
period = config.period
tau_A = config.tau_A
time_resolution = config.time_resolution
dt = config.dt
dphi = 0.01

k_b = 3.6
shift = -np.pi / 10
v_ext = -float(sys.argv[1])


# create functions effective_force_{q,a}_{l,r}
equilibrium_distance = 0
# (realistic_distance_function, equilibrium_distance) = basal.order_approximation(
#         second_order=0)
distance_function = basal.Deriver(basal.distance_approximation)
effective_forces = basal.generate_all_force_functions(
    distance_function, equilibrium_distance,
    equilibrium_distance, k_b)


friction_func = dyn.include_efficiency_dissipation(
    dyn.read_friction_matrix()[0])

initial_conditions = dyn.Q([0, dphi, 1, 1, 0, 0, 0])
(q_phi_l, q_phi_r, q_amp_l, q_amp_r) = drivers.forces_from_calibration(
        friction_func, omega)


driver = dyn.Driver(
    dyn.Q([False, False, False, False, True, True, True]),
    dyn.Q([lambda q: q_phi_l(q.phase_left) + effective_forces['p_l'](
               q.phase_left + shift, q.amplitude_left,
               q.phase_right + shift, q.amplitude_right),
           lambda q: q_phi_r(q.phase_right) + effective_forces['p_r'](
               q.phase_left + shift, q.amplitude_left,
               q.phase_right + shift, q.amplitude_right),
           lambda q: dyn.bordered_stiffness(
               q.phase_left, q.amplitude_left,
               q_amp_l, friction_func, tau_A, omega) + effective_forces['a_l'](
                   q.phase_left + shift, q.amplitude_left,
                   q.phase_right + shift, q.amplitude_right),
           lambda q: dyn.bordered_stiffness(
               q.phase_right, q.amplitude_right,
               q_amp_r, friction_func, tau_A, omega) + effective_forces['a_r'](
                   q.phase_left + shift, q.amplitude_left,
                   q.phase_right + shift, q.amplitude_right),
           dyn.const(0),
           dyn.const(v_ext),
           dyn.const(0)]))

(positions, velocities, forces) = dyn.beat_with_forces(
    initial_conditions, driver, dt, 7 * time_resolution,
    friction_func=friction_func)


np.savetxt('positions_{0}'.format(v_ext), positions.to_array())
np.savetxt('velocities_{0}'.format(v_ext), velocities.to_array())
np.savetxt('forces_{0}'.format(v_ext), forces.to_array())
