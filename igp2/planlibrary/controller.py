from collections import deque

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AdaptiveCruiseControl:
    """ Defines an adaptive cruise controller based on the intelligent driver model (IDM)"""

    def __init__(self, dt=0.05, a_a=5, b_a=5, delta=4., s_0=2., T_a=1.5):
        """ Initialise the parameters of the adaptive cruise controller

        Args:
            dt: temporal difference
            a_a: maximum positive acceleration
            b_a: maximum negative acceleration
            delta: acceleration exponent
            s_0: minimum desired gap
            T_a: following time-gap
        """
        self.dt = dt
        self.delta = delta
        self.s_0 = s_0
        self.a_a = a_a
        self.b_a = b_a
        self.T_a = T_a

    def get_acceleration(self, v_0: float, v_a: float, v_f: float, s_a: float) -> float:
        """ Get the acceleration output by the controller

        Args:
            v_0: maximum velocity
            v_a: ego vehicle velocity
            v_f: front vehicle velocity
            s_a: gap between vehicles

        Returns:
            acceleration
        """
        delta_v = v_a - v_f
        s_star = self.s_0 + self.T_a * v_a + v_a * delta_v / (2 * np.sqrt(self.a_a * self.b_a))
        accel = self.a_a * (1 - (v_a / v_0) ** self.delta - (s_star / s_a) ** 2)
        return accel * self.dt


class PIDController:
    """ PID controller based on the CARLA PID controller implementation. """

    def __init__(self,
                 dt: float = 0.05,
                 args_lateral: Dict[str, float] = None,
                 args_longitudinal: Dict[str, float] = None,
                 max_steering=1.0):
        """
        Constructor method.

        Args:
            dt: Discreet temporal difference
            args_lateral: dictionary of arguments to set the lateral PID controller
                using the following semantics:
                    K_P -- Proportional term
                    K_D -- Differential term
                    K_I -- Integral term
            args_longitudinal: dictionary of arguments to set the longitudinal
                PID controller using the following semantics:
                    K_P -- Proportional term
                    K_D -- Differential term
                    K_I -- Integral term
        """
        if args_lateral is None:
            args_lateral = {'K_P': 1.95, 'K_I': 0.2, 'K_D': 0.0}
        if args_longitudinal is None:
            args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0}
        self.max_steer = max_steering

        self.past_steering = 0.0
        self._lon_controller = PIDLongitudinalController(dt=dt, **args_longitudinal)
        self._lat_controller = PIDLateralController(dt=dt, **args_lateral)

    def next_action(self,
                    target_acceleration,
                    target_steering) -> (float, float):
        """Execute one step of control invoking both lateral and longitudinal
            PID controllers to reach a target waypoint
            at a given target_speed.

        Args:
            target_acceleration: Desired acceleration.
            target_steering: Target steering

        Returns:
            Acceleration and steering to execute
        """

        acceleration = self._lon_controller.next_action(target_acceleration)
        current_steering = self._lat_controller.next_action(target_steering)

        # Breaking adjustment: breaking is more sudden than speeding up
        if acceleration < 0:
            acceleration *= 2.

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.33:
            current_steering = self.past_steering + 0.33
        elif current_steering < self.past_steering - 0.33:
            current_steering = self.past_steering - 0.33
        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)
        self.past_steering = steering

        return acceleration, steering


class PIDLongitudinalController:
    """ PIDLongitudinalController implements longitudinal control using a PID. """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.05):
        """ Constructor method.

        Args:
            K_P: Proportional term
            K_D: Differential term
            K_I: Integral term
            dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def next_action(self, acceleration: float):
        """
        Execute one step of longitudinal control to reach a given target speed.

        Args:
            acceleration: Target acceleration

        Returns:
            acceleration control
        """
        self._error_buffer.append(acceleration)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return (self._k_p * acceleration) + (self._k_d * _de) + (self._k_i * _ie)


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.05):
        """ Constructor method.

        Args:
            K_P: Proportional term
            K_D: Differential term
            K_I: Integral term
            dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def next_action(self, steering):
        """Execute one step of lateral control to steer the vehicle towards a certain waypoint.

        Args:
            steering: Target steering

        Returns:
            steering control
        """
        self._e_buffer.append(steering)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        return (self._k_p * steering) + (self._k_d * _de) + (self._k_i * _ie)
