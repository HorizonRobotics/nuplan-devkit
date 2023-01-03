from typing import List

import numpy as np
import torch

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot
from nuplan.common.actor_state.state_representation import (StateSE2,
                                                            StateVector2D,
                                                            TimePoint)
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.simulation.controller.utils import \
    forward_integrate_tensor
from nuplan.planning.training.modeling.closed_loop_utilities.utils.torch_util import \
    principal_value


class BatchedKinematicBicycleModel(object):
    """
    A re-implementation of KinematicBicycleModel for PyTorch tensors.

    Original file: nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py
    """

    def __init__(
        self,
        vehicle: VehicleParameters,
        max_steering_angle: float = np.pi / 3,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ):
        """Construct BatchedKinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant
        self._device = None

    def initialize(self, **kwargs):
        """Initialize the motion model for a particular device."""
        self._device = kwargs["device"]

    def get_state_dot(self, state: List[EgoState]) -> List[EgoStateDot]:
        """Inherited, see super class."""
        longitudinal_speed = torch.tensor(
            [i.dynamic_car_state.rear_axle_velocity_2d.x for i in state],
            device=self._device,
        )
        heading = torch.tensor(
            [i.rear_axle.heading for i in state], device=self._device
        )
        steering_angle = torch.tensor(
            [i.tire_steering_angle for i in state], device=self._device
        )
        x_dot = longitudinal_speed * torch.cos(heading)
        y_dot = longitudinal_speed * torch.sin(heading)
        yaw_dot = (
            longitudinal_speed
            * torch.tan(steering_angle)
            / self._vehicle.wheel_base
        )

        ego_state_dot = []
        for x_dot_i, y_dot_i, yaw_dot_i, state_i in zip(
            x_dot, y_dot, yaw_dot, state
        ):
            ego_state_dot.append(
                EgoStateDot.build_from_rear_axle(
                    rear_axle_pose=StateSE2(
                        x=x_dot_i.item(),
                        y=y_dot_i.item(),
                        heading=yaw_dot_i.item(),
                    ),
                    rear_axle_velocity_2d=state_i.dynamic_car_state.rear_axle_acceleration_2d,
                    rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                    tire_steering_angle=state_i.dynamic_car_state.tire_steering_rate,
                    time_point=state_i.time_point,
                    is_in_auto_mode=True,
                    vehicle_parameters=self._vehicle,
                )
            )
        return ego_state_dot

    def _update_commands(
        self,
        state: List[EgoState],
        ideal_dynamic_state: List[DynamicCarState],
        sampling_time: List[TimePoint],
    ) -> List[EgoState]:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: a list of EgoState objects
        :param ideal_dynamic_state: a list of the desired dynamic state for propagation
        :param sampling_time: a list of time duration to propagate for
        :return: a list of propagating_state including updated dynamic_state
        """
        dt_control = torch.tensor(
            [i.time_s for i in sampling_time], device=self._device
        )
        accel = torch.tensor(
            [i.dynamic_car_state.rear_axle_acceleration_2d.x for i in state],
            device=self._device,
        )
        steering_angle = torch.tensor(
            [i.tire_steering_angle for i in state], device=self._device
        )

        ideal_accel = torch.tensor(
            [i.rear_axle_acceleration_2d.x for i in ideal_dynamic_state],
            device=self._device,
        )
        ideal_tire_steering_rate = torch.tensor(
            [i.tire_steering_rate for i in ideal_dynamic_state],
            device=self._device,
        )
        ideal_steering_angle = (
            dt_control * ideal_tire_steering_rate + steering_angle
        )

        updated_accel = (
            dt_control
            / (dt_control + self._accel_time_constant)
            * (ideal_accel - accel)
            + accel
        )
        updated_steering_angle = (
            dt_control
            / (dt_control + self._steering_angle_time_constant)
            * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        updated_steering_rate = (
            updated_steering_angle - steering_angle
        ) / dt_control

        dynamic_state = []
        for state_i, updated_accel_i, updated_steering_rate_i in zip(
            state, updated_accel, updated_steering_rate
        ):
            dynamic_state.append(
                DynamicCarState.build_from_rear_axle(
                    rear_axle_to_center_dist=state_i.car_footprint.rear_axle_to_center_dist,
                    rear_axle_velocity_2d=state_i.dynamic_car_state.rear_axle_velocity_2d,
                    rear_axle_acceleration_2d=StateVector2D(
                        updated_accel_i.item(), 0
                    ),
                    tire_steering_rate=updated_steering_rate_i.item(),
                )
            )
        propagating_state = []
        for state_i, dynamic_state_i in zip(state, dynamic_state):
            propagating_state.append(
                EgoState(
                    car_footprint=state_i.car_footprint,
                    dynamic_car_state=dynamic_state_i,
                    tire_steering_angle=state_i.tire_steering_angle,
                    is_in_auto_mode=True,
                    time_point=state_i.time_point,
                )
            )
        return propagating_state

    def propagate_state(
        self,
        state: List[EgoState],
        ideal_dynamic_state: List[DynamicCarState],
        sampling_time: List[TimePoint],
    ) -> None:
        """Inherited, see super class."""
        propagating_state: List[EgoState] = self._update_commands(
            state, ideal_dynamic_state, sampling_time
        )

        # Compute state derivatives
        state_dot: list[EgoStateDot] = self.get_state_dot(propagating_state)

        # NOTE: the following computations are torch Tensor based.

        # Integrate position and heading
        next_x = forward_integrate_tensor(
            [i.rear_axle.x for i in propagating_state],
            [i.rear_axle.x for i in state_dot],
            sampling_time,
            self._device,
        )
        next_y = forward_integrate_tensor(
            [i.rear_axle.y for i in propagating_state],
            [i.rear_axle.y for i in state_dot],
            sampling_time,
            self._device,
        )
        next_heading = forward_integrate_tensor(
            [i.rear_axle.heading for i in propagating_state],
            [i.rear_axle.heading for i in state_dot],
            sampling_time,
            self._device,
        )
        # Wrap angle between [-pi, pi]
        next_heading = principal_value(next_heading)

        # Compute rear axle velocity in car frame
        next_point_velocity_x = forward_integrate_tensor(
            [
                i.dynamic_car_state.rear_axle_velocity_2d.x
                for i in propagating_state
            ],
            [i.dynamic_car_state.rear_axle_velocity_2d.x for i in state_dot],
            sampling_time,
            self._device,
        )
        next_point_velocity_y = (
            0.0  # Lateral velocity is always zero in kinematic bicycle model
        )

        # Integrate steering angle and clip to bounds
        next_point_tire_steering_angle = torch.clip(
            forward_integrate_tensor(
                [i.tire_steering_angle for i in propagating_state],
                [i.tire_steering_angle for i in state_dot],
                sampling_time,
                self._device
            ),
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        # Compute angular velocity
        next_point_angular_velocity = (
            next_point_velocity_x
            * torch.tan(next_point_tire_steering_angle)
            / self._vehicle.wheel_base
        )

        rear_axle_accel = [
            [
                i.dynamic_car_state.rear_axle_velocity_2d.x,
                i.dynamic_car_state.rear_axle_velocity_2d.y,
            ]
            for i in state_dot
        ]

        angular_velocity_tensor = torch.tensor(
            [i.dynamic_car_state.angular_velocity for i in state],
            device=self._device,
        )
        sampling_time_tensor = torch.tensor(
            [i.time_s for i in sampling_time], device=self._device
        )

        angular_accel = (
            next_point_angular_velocity - angular_velocity_tensor
        ) / sampling_time_tensor

        ego_state = []
        for (
            next_x_i,
            next_y_i,
            next_heading_i,
            next_point_velocity_x_i,
            rear_axle_accel_i,
            next_point_tire_steering_angle_i,
            sampling_time_i,
            next_point_angular_velocity_i,
            angular_accel_i,
            state_dot_i,
            propagating_state_i,
        ) in zip(
            next_x,
            next_y,
            next_heading,
            next_point_velocity_x,
            rear_axle_accel,
            next_point_tire_steering_angle,
            sampling_time,
            next_point_angular_velocity,
            angular_accel,
            state_dot,
            propagating_state,
        ):
            ego_state.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(
                        next_x_i.item(), next_y_i.item(), next_heading_i.item()
                    ),
                    rear_axle_velocity_2d=StateVector2D(
                        next_point_velocity_x_i.item(), next_point_velocity_y
                    ),
                    rear_axle_acceleration_2d=StateVector2D(
                        rear_axle_accel_i[0], rear_axle_accel_i[1]
                    ),
                    tire_steering_angle=float(
                        next_point_tire_steering_angle_i.item()
                    ),
                    time_point=propagating_state_i.time_point
                    + sampling_time_i,
                    vehicle_parameters=self._vehicle,
                    is_in_auto_mode=True,
                    angular_vel=next_point_angular_velocity_i.item(),
                    angular_accel=angular_accel_i.item(),
                    tire_steering_rate=state_dot_i.tire_steering_angle,
                )
            )
        return ego_state
