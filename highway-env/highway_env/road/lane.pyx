from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional
import numpy as np

from highway_env import utils
from highway_env.types import Vector
# from highway_env.utils import wrap_to_pi

DEFAULT_WIDTH = 4
VEHICLE_LENGTH = 5

# cdef class AbstractLane(object):
class AbstractLane(object):

    """A lane on the road, described by its central curve."""

    # def __cinit__(self):
    # # metaclass__ = ABCMeta
        # self.DEFAULT_WIDTH = 4
        # self.VEHICLE_LENGTH = 5
        # self.length = 0
    # # line_types: Tuple["LineType"]

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    # @profile
    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) \
            -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = abs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
            -VEHICLE_LENGTH <= longitudinal < self.length + VEHICLE_LENGTH
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and \
            0 <= longitudinal < self.length + VEHICLE_LENGTH
        return is_close

    def after_end(self, position: np.ndarray, longitudinal: np.float64 = None, lateral: float = None, vehicle_length: float = 5) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - vehicle_length
        # return longitudinal > self.length - (10 + VEHICLE_LENGTH / 2)

    def distance_to_end(self, position: np.ndarray, longitudinal: np.float64 = None, lateral: float = None) -> bool:
        """Compute distance from position to the end of the lane"""
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return self.length - longitudinal

    def distance(self, position: np.ndarray):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_between_points(self, position1: np.ndarray, position2: np.ndarray):
        """Compute the lane distance between two points on the lane"""
        return self.local_coordinates(position2)[0] - self.local_coordinates(position1)[0]

    # @profile
    # cpdef float distance_with_heading(self, position: np.ndarray, heading: Optional[float]):
    def distance_with_heading(self, position: np.ndarray, heading: Optional[float]):
        """Compute a weighted distance in position and heading to the lane."""
        # if heading is None:
            # return self.distance(position)
        # s, r = self.local_coordinates(position)
        # # angle = abs(wrap_to_pi(heading - self.heading_at(s)))
        # return abs(r) + max(s - self.length, 0) + max(0 - s, 0) #+ heading_weight*angle

        # print("called distance with heading on abstract lane")

        cdef float s, r, length
        coords = self.local_coordinates(position)
        s = coords[0]
        r = coords[1]
        length = self.length
        return abs(r) + max(s - length, 0) + max( -s, 0) #+ heading_weight*angle

class LineType:

    """A lane side line type."""

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class StraightLane(AbstractLane):

    """A lane going in straight line."""

    def __init__(self,
                 start: Vector,
                 end: Vector,
                 width: float = DEFAULT_WIDTH,
                 line_types: Tuple[LineType, LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for determining who has right of way
        """
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.width = width
        self.heading = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.length = float(np.linalg.norm(self.end - self.start))
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit

        # print(f"{self.direction=} {self.direction_lateral=}")

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    # @profile
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.start
        _dir = self.direction
        _dir_lat = self.direction_lateral

        cdef float dx, dy, dir_x, dir_y, dir_lat_x, dir_lat_y
        dx = delta[0]
        dy = delta[1]
        dir_x = _dir[0]
        dir_y = _dir[1]
        dir_lat_x = _dir_lat[0]
        dir_lat_y = _dir_lat[1]

        longitudinal = dx * dir_x + dy * dir_y
        lateral  = dx * dir_lat_x + dy * dir_lat_y

        # longitudinal = np.dot(delta, self.direction)
        # lateral = np.dot(delta, self.direction_lateral)
        # print(f"{self.__class__.__name__}{longitudinal=}{lateral=}")
        return longitudinal, lateral

class HorizontalLane(StraightLane):
    """A lane only going horizontally on the screen"""

    def __init__(self,
                 start: Vector,
                 end: Vector,
                 width: float = DEFAULT_WIDTH,
                 line_types: Tuple[LineType, LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:

        # make sure its actually a horizontal lane
        assert(start[1] == end[1])
        super().__init__(start, end,  width, line_types, forbidden, speed_limit, priority)

        # self.cos_heading = np.cos(self.heading)
        # self.sin_heading = np.sin(self.heading)

        self.vec = self.end - self.start
        self.norm_vec = np.linalg.norm(self.vec)


    # @profile
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        return position[0] - self.start[0], position[1] - self.start[1]

    # @profile
    def distance_with_heading(self, position: np.ndarray, heading: Optional[float]):
        """Compute a weighted distance in position and heading to the lane."""
        # if heading is None:
            # return self.distance(position)
        cdef float s, r, length
        # s, r = self.local_coordinates(position)
        # angle = abs(wrap_to_pi(heading))

        # print("called distance with heading on horizontal lane")

        d = position - self.start
        s = d[0]
        r = d[1]
        length = self.length

        # return abs(r) + max(s - self.length, 0) + max(-s, 0) #+ heading_weight*angle
        return abs(r) + max(s - length, 0) + max(-s, 0) #+ heading_weight*angle

        ## Version only works for infinite lines, not line segments :(
        # a = self.vec
        # b = position-self.start
        # # fast hacky cross product
        # c = a[0]*b[1] - a[1]*b[0]

        # # return np.cross(self.vec, position-self.start) / self.norm_vec
        # return c / self.norm_vec

        # Version 3
        # d = position - self.start
        # return np.abs(d[1]) + np.max([-d[0], d[0] - self.norm_vec])
        # return 0
        # return d[1] + np.max([-d[0], d[0] - self.norm_vec])
        # return np.sum([np.abs(d[1]),  np.max([-d[0], d[0] - self.norm_vec])])



class SineLane(StraightLane):

    """A sinusoidal lane."""

    def __init__(self,
                 start: Vector,
                 end: Vector,
                 amplitude: float,
                 pulsation: float,
                 phase: float,
                 width: float = DEFAULT_WIDTH,
                 line_types: Tuple[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        """
        New sinusoidal lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        """
        super().__init__(start, end,  width, line_types, forbidden, speed_limit, priority)
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return super().position(longitudinal,
                                lateral + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase))

    def heading_at(self, longitudinal: float) -> float:
        return super().heading_at(longitudinal) + np.arctan(
            self.amplitude * self.pulsation * np.cos(self.pulsation * longitudinal + self.phase))

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        longitudinal, lateral = super().local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(self.pulsation * longitudinal + self.phase)


class CircularLane(AbstractLane):

    """A lane going in circle arc."""

    def __init__(self,
                 center: Vector,
                 radius: float,
                 start_phase: float,
                 end_phase: float,
                 clockwise: bool = True,
                 width: float = DEFAULT_WIDTH,
                 line_types: Tuple[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.length = radius*(end_phase - start_phase) * self.direction
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction)*np.array([np.cos(phi), np.sin(phi)])

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi/2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + utils.wrap_to_pi(phi - self.start_phase)
        r = np.linalg.norm(delta)
        longitudinal = self.direction*(phi - self.start_phase)*self.radius
        lateral = self.direction*(self.radius - r)
        return longitudinal, lateral

    def distance_between_points(self, position1: np.ndarray, position2: np.ndarray):
        delta1 = position1 - self.center
        phi1 = np.arctan2(delta1[1], delta1[0])

        delta2 = position2 - self.center
        phi2 = np.arctan2(delta2[1], delta2[0])

        phi = utils.wrap_to_pi(phi2 - phi1)

        return self.radius * phi

    def distance_to_end(self, position: np.ndarray, longitudinal: np.float64 = None, lateral: float = None) -> bool:
        """Compute distance from position to the end of the lane"""
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])

        phi = utils.wrap_to_pi(self.end_phase - phi)

        return self.radius * phi

    def distance_with_heading(self, position: np.ndarray, heading: Optional[float]):
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])

        in_range = (phi-self.start_phase) % (2*np.pi) <= (self.end_phase - self.start_phase) % (2*np.pi)
        # print(f"{in_range=}")
        if in_range:
            val = abs(np.linalg.norm(delta) - self.radius)
            # print("on lane")
            return val
        else:
            start = self.position(0, 0)
            end = self.position(self.length, 0)
            val =  min(np.linalg.norm(position - start), np.linalg.norm(position-end))
            # print("not on lane")
            return val

