from typing import Tuple, Union

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.types import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import RoadObject


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    """polite behavior"""
    # """Longitudinal policy parameters"""
    # Maximum acceleration.
    ACC_MAX = 6.0  # [m/s2]
    # Desired maximum acceleration.
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    # COMFORT_ACC_MAX = 0.3  # [m/s2]
    # Desired maximum deceleration.
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    # COMFORT_ACC_MIN = -3.0  # [m/s2]
    # Desired jam distance to the front vehicle.
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    # Desired time gap to the front vehicle.
    TIME_WANTED = 1.5  # [s]
    # TIME_WANTED = 1.7  # [s]
    # Exponent of the velocity term.
    DELTA = 4.0  # []

    """Lateral policy parameters"""
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 9.0  # [m/s2]
    # LANE_CHANGE_MAX_BRAKING_IMPOSED = 40.0  # [m/s2]
    # LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    RIGHT_BIAS = 0.0  # bias for lane changes to the right

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        config=None,
    ):
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            config,
        )
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.duTactical = 200
        self.exit_lane = self.road.network.get_lane(("c", "d", 1))

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
            timer=getattr(vehicle, "timer", None),
        )
        return v

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.surrounding_vehicles(self)
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = utils.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        distance_to_exit = self.exit_lane.distance(self.position)

        # only decelearte if we are on the wrong lane
        if not self.on_track():
            self.alpha_v0 = max(0.4, distance_to_exit / self.duTactical)
        else:  # reset after passing exit
            self.alpha_v0 = 1

        # currently lane change happening
        if self.target_lane_index != self.lane_index:
            front_vehicle, _ = self.road.surrounding_vehicles(
                self, self.target_lane_index
            )

        # Longitudinal: IDM
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        action["acceleration"] = utils.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        Vehicle.act(
            self, action
        )  # Skip ControlledVehicle.act(), or the command will be overriden.

    def on_track(self):
        if not (self.lane_index[0] == "b" or self.lane_index[1] == "c"):
            return True
        if (
            self.lane_index == ("b", "c", 0) or self.lane_index == ("b", "c", 1)
        ) and self.RIGHT_BIAS < -0.01:
            return True
        elif self.lane_index == (
            "b",
            "c",
            2,
        ):  # Merging vehicles #and self.RIGHT_BIAS > 0.1:
            return True
        else:
            return False

    def step(self, dt: float):
        """
        Step the simulation.
        Increases a timer used for decision policies, and step the vehicle dynamics.
        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or isinstance(ego_vehicle, RoadObject):
            return 0
        ego_target_speed = utils.not_zero(getattr(ego_vehicle, "target_speed", 0))

        # adjust target speed for special circumstances
        ego_target_speed *= ego_vehicle.alpha_v0

        acceleration = self.COMFORT_ACC_MAX * (
            1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA)
        )


        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )

        return acceleration

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = False,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def maximum_speed(self, front_vehicle: Vehicle = None) -> Tuple[float, float]:
        """
        Compute the maximum allowed speed to avoid Inevitable Collision States.

        Assume the front vehicle is going to brake at full deceleration and that
        it will be noticed after a given delay, and compute the maximum speed
        which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed speed, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_speed
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(
            self.lane_distance_to(front_vehicle)
            - self.LENGTH / 2
            - front_vehicle.LENGTH / 2
            - d0,
            0,
        )
        v1_0 = front_vehicle.speed
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1**2) * d + 4 * a0 * a1 * v1_0**2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Speed control
        self.target_speed = min(self.maximum_speed(front_vehicle), self.target_speed)
        acceleration = self.speed_control(self.target_speed)

        return v_max, acceleration

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # Only allow this lane change if mobil model allows it
            if not self.mobil(self.target_lane_index):
                self.target_lane_index = self.lane_index

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(
                self.position
            ):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.surrounding_vehicles(self, lane_index)
        old_preceding, old_following = self.road.surrounding_vehicles(self)

        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )

        # unsafe braking required?
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
        old_following_a = self.acceleration(
            ego_vehicle=old_following, front_vehicle=self
        )
        old_following_pred_a = self.acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding
        )
        jerk = (
            self_pred_a
            - self_a
            + self.POLITENESS
            * (
                new_following_pred_a
                - new_following_a
                + old_following_pred_a
                - old_following_a
            )
        )

        if self.lane_index[2] > lane_index[2]:  # change to right lane
            bias = self.RIGHT_BIAS
        elif self.lane_index[2] < lane_index[2]:  # change to left lane
            bias = -self.RIGHT_BIAS
        else:
            bias = 0

        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN + bias:
            return False

        # All clear, let's go!
        return True

