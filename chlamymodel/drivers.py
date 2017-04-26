"""
this is a collection of commonly used drivers and friction matrices
"""
import numpy as np
import chlamymodel.seven as dyn


def constant_efficiency_friction(efficiency=0.2):
    """ consider internal friction, proportional to active force """
    return dyn.include_efficiency_dissipation(
        dyn.read_friction_matrix()[0],
        efficiency_ratio=efficiency,
        amplitude_efficiency=True)


def constant_intraflagellar_friction(efficiency=0.2):
    """ consider internal friction as a constant contribution """
    friction_func = dyn.read_friction_matrix()[0]
    return dyn.include_constant_phase_dissipation(
        friction_func, efficiency_ratio=efficiency)


def calibration_driver(omega):
    """ used for calibrating. forces are obtained from it.  """
    return dyn.Driver(
        dyn.Q([True, True, True, True, True, True, True]),
        dyn.Q([dyn.const(omega),
               dyn.const(omega),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0)]))


def clamped_driver(
        driving_forces, friction_func, omega, tau_A, v_ext=dyn.const(0),
        phase_force_addition=(dyn.const(0), dyn.const(0))):
    """
    a clamped cell.
    """
    (q_phi_l, q_phi_r, q_amp_l, q_amp_r) = driving_forces
    return dyn.Driver(
        dyn.Q([False, False, False, False, True, True, True]),
        dyn.Q([lambda q: q_phi_l(q.phase_left) + phase_force_addition[0](q),
               lambda q: q_phi_r(q.phase_right) + phase_force_addition[1](q),
               lambda q: dyn.bordered_stiffness(
                   q.phase_left, q.amplitude_left,
                   q_amp_l, friction_func, tau_A, omega),
               lambda q: dyn.bordered_stiffness(
                   q.phase_right, q.amplitude_right,
                   q_amp_r, friction_func, tau_A, omega),
               dyn.const(0),
               v_ext,
               dyn.const(0)]))


def clamped_driver_without_amplitude(
        driving_forces, friction_func, omega, v_ext=dyn.const(0),
        phase_force_addition=(dyn.const(0), dyn.const(0))):
    """
    a clamped cell.
    """
    (q_phi_l, q_phi_r, q_amp_l, q_amp_r) = driving_forces
    return dyn.Driver(
        dyn.Q([False, False, True, True, True, True, True]),
        dyn.Q([lambda q: q_phi_l(q.phase_left) + phase_force_addition[0](q),
               lambda q: q_phi_r(q.phase_right) + phase_force_addition[1](q),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0),
               v_ext,
               dyn.const(0)]))


def elastically_clamped_driver(
        driving_forces, friction_func, omega, tau_A, k_a):
    """ clamping via force """
    (q_phi_l, q_phi_r, q_amp_l, q_amp_r) = driving_forces
    return dyn.Driver(
        dyn.Q([False, False, False, False, True, True, False]),
        dyn.Q([lambda q: q_phi_l(q.phase_left),
               lambda q: q_phi_r(q.phase_right),
               lambda q: dyn.bordered_stiffness(
                   q.phase_left, q.amplitude_left,
                   q_amp_l, friction_func, tau_A, omega),
               lambda q: dyn.bordered_stiffness(
                   q.phase_right, q.amplitude_right,
                   q_amp_r, friction_func, tau_A, omega),
               dyn.const(0),
               dyn.const(0),
               lambda q: -k_a * q.orientation]))


def free_driver(driving_forces, friction_func, omega, tau_A,
                phase_force_addition=(dyn.const(0), dyn.const(0))):
    """ freely swimming chlamy """
    (q_phi_l, q_phi_r, q_amp_l, q_amp_r) = driving_forces
    return dyn.Driver(
        dyn.Q([False, False, False, False, False, False, False]),
        dyn.Q([lambda q: q_phi_l(q.phase_left) + phase_force_addition[0](q),
               lambda q: q_phi_r(q.phase_right) + phase_force_addition[1](q),
               lambda q: dyn.bordered_stiffness(
                   q.phase_left, q.amplitude_left,
                   q_amp_l, friction_func, tau_A, omega),
               lambda q: dyn.bordered_stiffness(
                   q.phase_right, q.amplitude_right,
                   q_amp_r, friction_func, tau_A, omega),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0)]))


def free_driver_without_amplitude(driving_forces, friction_func, omega,
                                  phase_force_addition=(dyn.const(0), dyn.const(0))):
    (q_phi_l, q_phi_r, q_amp_l, q_amp_r) = driving_forces
    return dyn.Driver(
        dyn.Q([False, False, True, True, False, False, False]),
        dyn.Q([lambda q: q_phi_l(q.phase_left) + phase_force_addition[0](q),
               lambda q: q_phi_r(q.phase_right) + phase_force_addition[1](q),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0),
               dyn.const(0)]))


situation_drivers = {
    'susceptibilities': lambda d, f, o, t: clamped_driver(d, f, o, t, v_ext=dyn.const(-1)),
    'nonisochrony': free_driver,
    'squareload': lambda d, f, o, t, v: clamped_driver(d, f, o, t, v_ext=v),
    'free': free_driver,
    'clamped': lambda d, f, o, t: clamped_driver(d, f, o, t, v_ext=dyn.const(0))}


def forces_from_calibration(friction, omega):
    """ model calibration for obtaining generalized forces """
    time_resolution = 100
    frequency = omega / 2 / np.pi
    dt = (1 / frequency) / time_resolution
    initial_condition = dyn.Q([0, 0, 1, 1, 0, 0, 0])

    (_, _, forces) = dyn.beat_with_forces(
        initial_condition, calibration_driver(omega),
        dt, time_resolution, friction_func=friction)

    return [dyn.make_phi_func(d) for d in [forces.phase_left,
                                           forces.phase_right,
                                           forces.amplitude_left,
                                           forces.amplitude_right]]


def a_model(friction_func, tau_A):
    """
    the model is defined by how internal friction is treated and the phase
    dependence of the amplitude correlation

    return a function, that when called returns a driver. for the squareload
    function one additional argument has to be provided.
    """
    def inner(situation, omega, *args):
        """ the driver depends not only on tau, but also on the experiment """
        friction = friction_func(efficiency=0.2)
        generalized_forces = forces_from_calibration(friction, omega)
        driver = situation_drivers[situation](
            generalized_forces, friction, omega, tau_A, *args)
        return (friction, driver)
    return inner


tau_constant = 5.9
tau_phase = lambda phi: 5.9 * (1 + 0.5 * np.sin(phi))


models = [a_model(constant_efficiency_friction, tau_constant),
          a_model(constant_intraflagellar_friction, tau_constant),
          a_model(constant_efficiency_friction, tau_phase),
          a_model(constant_intraflagellar_friction, tau_phase)]



def one_synchronous_trajectory(friction, driver, omega, step_number):
    """ do the synchronous beat for a while """
    dt = 2 * np.pi / omega / 100
    initial_condition = dyn.Q([0, 0, 1, 1, 0, 0, 0])
    return dyn.beat_with_forces(
        initial_condition, driver, dt, step_number, friction_func=friction,
        symmetric=True)


def one_asynchronous_trajectory(friction, driver, omega, dphi, step_number):
    """ do an asynchronous beat for a while """
    dt = 2 * np.pi / omega / 100
    initial_condition = dyn.Q([0, dphi, 1, 1, 0, 0, 0])
    return dyn.beat_with_forces(
        initial_condition, driver, dt, step_number, friction_func=friction)


def limit_cycle_stability(friction, driver, omega, n_phase=18, n_amplitude=11):
    """ do many calculations and obtain velocities """

    dt = 2 * np.pi / omega / 100
    result = []

    def one_dataset(phase, amplitude):
        """ obtain velocities for certain phase space point """
        initial_condition = dyn.Q([phase, phase, amplitude, amplitude, 0, 0, 0])
        (positions, velocities, _) = dyn.beat_with_forces(
            initial_condition, driver, dt, 1, friction_func=friction)
        return (positions.phase_left[0], positions.amplitude_left[0],
                velocities.phase_left[0], velocities.amplitude_left[0])

    for phi in np.linspace(0, 2 * np.pi, n_phase + 1)[:-1]:
        for amp in np.linspace(0.8, 1.2, n_amplitude):
            result.append(one_dataset(phi, amp))

    return result


omega = 2 * np.pi * 0.05


def get_susceptibilities(model):
    """ susceptibilities """
    (friction, driver) = models[model]('susceptibilities', omega)
    return one_synchronous_trajectory(friction, driver, omega, 300)


def get_nonisochrony(model):
    """ non isochrony """
    (friction, driver) = models[model]('nonisochrony', omega)
    return limit_cycle_stability(friction, driver, omega,
                                 n_phase=18, n_amplitude=13)

def get_free_synch(model):
    """ synchronization strength of the free swimmer """
    (friction, driver) = models[model]('free', omega)
    return one_asynchronous_trajectory(friction, driver, omega, 0.01, 500)


def get_clamped_synch(model):
    """ synchronization strength of clamped chlamy """
    (friction, driver) = models[model]('clamped', omega)
    return one_asynchronous_trajectory(friction, driver, omega, 0.01, 500)


def get_squareload(model, u_max, omega_factor):
    """ phase locking with external periodic flow """
    (friction, driver) = models[model](
        'susceptibilities', omega,
        lambda t: u_max * np.sign(np.sin(omega_factor * omega * t)))
    return one_synchronous_trajectory(friction, driver, omega, 5000)


if __name__ == '__main__':

    for model in range(4):

#         (pos, vel, frc) = get_susceptibilities(model)
#         np.savetxt('data/susc_{0}_pos'.format(model), pos.to_array())
#         np.savetxt('data/susc_{0}_vel'.format(model), vel.to_array())
#         np.savetxt('data/susc_{0}_frc'.format(model), frc.to_array())

#         result = get_nonisochrony(model)
#         np.savetxt('data/noni_{0}'.format(model), result)

#         (pos, vel, frc) = get_free_synch(model)
#         np.savetxt('data/free_synch_{0}_pos'.format(model), pos.to_array())
#         np.savetxt('data/free_synch_{0}_vel'.format(model), vel.to_array())
#         np.savetxt('data/free_synch_{0}_frc'.format(model), frc.to_array())

        (pos, vel, frc) = get_clamped_synch(model)
        np.savetxt('data/clamp_synch_{0}_pos'.format(model), pos.to_array())
        np.savetxt('data/clamp_synch_{0}_vel'.format(model), vel.to_array())
        np.savetxt('data/clamp_synch_{0}_frc'.format(model), frc.to_array())
