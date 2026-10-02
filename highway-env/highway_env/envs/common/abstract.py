import copy
import os
from typing import Tuple, Optional, Callable
from copy import deepcopy
import gymnasium as gym
import numpy as np

from highway_env import utils
from highway_env.envs.common.action import (
    action_factory,
    Action,
    ActionType,
)
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import Obstacle, Landmark

Observation = np.ndarray
DEFAULT_WIDTH: float = 4  # width of the straight lane


class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """

    observation_type: ObservationType
    action_type: ActionType
    automatic_rendering_callback: Optional[Callable]
    metadata = {"render.modes": ["human", "rgb_array"]}
    render_mode = "human"

    PERCEPTION_DISTANCE = 6.0 * MDPVehicle.SPEED_MAX
    """The maximum distance of any vehicle present in the observation [m]"""

    config = {}

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # Seeding
        self.np_random = None

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = "human"
        self.enable_auto_render = False

        self.ACTIONS_ALL = {
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4,
        }

        self.reset()

    def set_config(self, config_):
        print(f"setting external config to {config_}")
        self.config = config_

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                # "type": "Kinematics"
                "type": "LidarObservation"
            },
            "action": {"type": "DiscreteMetaAction"},
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 1200,  # [px]
            "screen_height": 500,  # [px]
            "centering_position": [0.6, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.

        If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self, seed=None, options=None) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        super().reset(seed=seed)

        # self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self.vehicle_speed = []
        self.vehicle_pos = []
        self._reset()
        # self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created

        # set the vehicle id for visualizing
        for i, v in enumerate(self.road.vehicles):
            v.id = i
        obs = self.observation_type.observe()

        self.road.initial_vehicles = deepcopy(self.road.vehicles)

        return obs, {}

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        average_speed = 0
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.steps += 1

        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        for v in self.controlled_vehicles:
            average_speed += v.speed
        average_speed = average_speed / len(self.controlled_vehicles)

        average_road_speed = 0
        for v in self.road.vehicles:
            average_road_speed += v.speed
        average_road_speed = average_road_speed / len(self.road.vehicles)

        self.vehicle_speed = [v.speed for v in self.controlled_vehicles]
        self.vehicle_pos = [v.position for v in self.controlled_vehicles]

        # did any other vehicle on the road crash?
        other_vehciles = filter(lambda v: v != self.vehicle, self.road.vehicles)
        crashes = [veh.crashed for veh in other_vehciles]

        # did the ego vehicle merge succesfully
        ego_veh_lane = self.road.network.get_closest_lane_index(
            self.controlled_vehicles[0].position, 0.0
        )
        merged = ego_veh_lane == ("c", "d", 0) or ego_veh_lane == ("c", "d", 1)

        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "other_crashes": any(crashes),
            "action": action,
            "average_speed": average_speed,
            "average_road_speed": average_road_speed,
            "vehicle_speed": self.vehicle_speed,
            "vehicle_position": self.vehicle_pos,
            "merged": merged,
        }

        # expose which of the radars perceives interference this timestep
        if hasattr(self.observation_type, "radars_affected_for_whole_timestep"):
            info["radars_affected_for_whole_timestep"] = (
                self.observation_type.radars_affected_for_whole_timestep.copy()
            )

        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return obs, reward, terminal, False, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(
            int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        ):
            # Forward action to the vehicle
            if (
                action is not None
                and not self.config["manual_control"]
                and self.time
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)  # defined in action.py

            self.road.act()  # Execute an action
            self.road.step(
                1 / self.config["simulation_frequency"]
            )  # propagate the vehicle state given its actions.
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == "rgb_array":
            image = self.viewer.get_image()
            return image
        self.should_update_rendering = False

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.

        If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
        such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback is not None:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def distance_to_merging_end(self, vehicle):
        distance_to_end = self.ends[2]
        if vehicle.lane_index == ("b", "c", 1):
            distance_to_end = sum(self.ends[:3]) - vehicle.position[0]
        return distance_to_end

    def _compute_headway_distance(
        self,
        vehicle,
    ):
        headway_distance = 60
        for v in self.road.vehicles:
            if (v.lane_index == vehicle.lane_index) and (
                v.position[0] > vehicle.position[0]
            ):
                hd = v.position[0] - vehicle.position[0]
                if hd < headway_distance:
                    headway_distance = hd

            # also consider the vehicles on the next road segmentation connected to the current lane
            if (
                (vehicle.lane_index != ("b", "c", 1))
                and (
                    v.lane_index
                    == self.road.network.next_lane(
                        vehicle.lane_index, position=vehicle.position
                    )
                )
                and (v.position[0] > vehicle.position[0])
            ):
                hd = v.position[0] - vehicle.position[0]
                if hd < headway_distance:
                    headway_distance = hd
        return headway_distance

    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ["viewer", "automatic_rendering_callback"]:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def check_collision(self, vehicle, other, other_trajectories):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle' trajectories or object
        other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]
        """
        if vehicle.crashed or other is vehicle:
            return

        if isinstance(other, Vehicle):
            if self._is_colliding(vehicle, other, other_trajectories):
                vehicle.speed = other_trajectories[2] = min(
                    [vehicle.speed, other_trajectories[2]], key=abs
                )
                vehicle.crashed = other.crashed = True

        elif isinstance(other, Obstacle):
            if self._is_colliding(vehicle, other, other_trajectories):
                vehicle.speed = min([vehicle.speed, 0], key=abs)
                vehicle.crashed = other.hit = True
        elif isinstance(other, Landmark):
            if self._is_colliding(vehicle, other, other_trajectories):
                other.hit = True

    def _is_colliding(self, vehicle, other, other_trajectories):
        # Fast spherical pre-check
        # other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]

        # Euclidean distance
        # if np.linalg.norm(other_trajectories[0] - vehicle.position) > vehicle.LENGTH:
        if utils.norm(other_trajectories[0], vehicle.position) > vehicle.LENGTH:
            return False

        # Accurate rectangular check
        return utils.rotated_rectangles_intersect(
            (
                vehicle.position,
                0.9 * vehicle.LENGTH,
                0.9 * vehicle.WIDTH,
                vehicle.heading,
            ),
            (
                other_trajectories[0],
                0.9 * other.LENGTH,
                0.9 * other.WIDTH,
                other_trajectories[1],
            ),
        )
