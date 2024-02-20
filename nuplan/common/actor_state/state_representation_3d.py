from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Union

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint


@dataclass
class Point3D:
    """Class to represents 3D points."""

    x: float  # [m] location
    y: float  # [m] location
    z: float  # [m] location
    __slots__ = "x", "y", "z"

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y, z)
        """
        return iter((self.x, self.y, self.z))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y, z]
        """
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y, self.z))


@dataclass
class StateSE3(Point3D):
    """
    SE3 state - representing [x, y, z, heading], we currently ignore pitch and roll
    """

    heading: float  # [rad] heading of a state
    __slots__ = "heading"

    @property
    def point(self) -> Point3D:
        """
        Gets a point from the StateSE3
        :return: Point with x, y, z from StateSE3
        """
        return Point3D(self.x, self.y, self.z)

    # def as_matrix(self) -> npt.NDArray[np.float32]:
    #     """
    #     :return: 3x3 2D transformation matrix representing the SE2 state.
    #     """
    #     return np.array(
    #         [
    #             [np.cos(self.heading), -np.sin(self.heading), self.x],
    #             [np.sin(self.heading), np.cos(self.heading), self.y],
    #             [0.0, 0.0, 1.0],
    #         ]
    #     )

    def as_matrix_3d(self) -> npt.NDArray[np.float32]:
        """
        :return: 4x4 3D transformation matrix representing the SE3 state, ignoring pitch and roll
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), 0.0, self.x],
                [np.sin(self.heading), np.cos(self.heading), 0.0, self.y],
                [0.0, 0.0, 1.0, self.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def distance_to(self, state: StateSE3) -> float:
        """
        Compute the euclidean distance between two points
        :param state: state to compute distance to
        :return distance between two points
        """
        return float(np.linalg.norm(np.array([self.x - state.x, self.y - state.y, self.z - state.z])))

    @staticmethod
    def from_matrix(matrix: npt.NDArray[np.float32]) -> StateSE3:
        """
        :param matrix: 3x3 2D transformation matrix
        :return: StateSE2 object
        """
        #TODO fix this to take 4x4 mat
        assert matrix.shape == (3, 3), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        vector = [matrix[0, 2], matrix[1, 2], np.arctan2(matrix[1, 0], matrix[0, 0])]
        return StateSE3.deserialize(vector)

    @staticmethod
    def deserialize(vector: List[float]) -> StateSE3:
        """
        Deserialize vector into state SE3
        :param vector: serialized list of floats
        :return: StateSE3
        """
        if len(vector) != 4:
            raise RuntimeError(f'Expected a vector of size 4, got {len(vector)}')

        return StateSE3(x=vector[0], y=vector[1], z=vector[2], heading=vector[2])

    def serialize(self) -> List[float]:
        """
        :return: list of serialized variables [X, Y, Z, Heading]
        """
        return [self.x, self.y, self.z, self.heading]

    def __eq__(self, other: object) -> bool:
        """
        Compare two state SE3
        :param other: object
        :return: true if the objects are equal, false otherwise
        """
        if not isinstance(other, StateSE3):
            # Return NotImplemented in case the classes are not of the same type
            return NotImplemented
        return (
            math.isclose(self.x, other.x, abs_tol=1e-3)
            and math.isclose(self.y, other.y, abs_tol=1e-3)
            and math.isclose(self.z, other.z, abs_tol=1e-3)
            and math.isclose(self.heading, other.heading, abs_tol=1e-4)
        )

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y, z, heading)
        """
        return iter((self.x, self.y, self.z, self.heading))

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash((self.x, self.y, self.z, self.heading))


@dataclass
class ProgressStateSE3(StateSE3):
    """
    StateSE3 parameterized by progress
    """

    progress: float  # [m] distance along a path
    __slots__ = "progress"

    @staticmethod
    def deserialize(vector: List[float]) -> ProgressStateSE3:
        """
        Deserialize vector into this class
        :param vector: containing raw float numbers containing [progress, x, y, z, heading]
        :return: ProgressStateSE3 class
        """
        if len(vector) != 5:
            raise RuntimeError(f'Expected a vector of size 5, got {len(vector)}')

        return ProgressStateSE3(progress=vector[0], x=vector[1], y=vector[2], z=vector[3], heading=vector[3])

    def __iter__(self) -> Iterable[Union[float]]:
        """
        :return: an iterator over the tuble of (progress, x, y, z, heading) states
        """
        return iter((self.progress, self.x, self.y, self.z, self.heading))


@dataclass
class TemporalStateSE3(StateSE3):
    """
    Representation of a temporal state
    """

    time_point: TimePoint  # state at a time

    @property
    def time_us(self) -> int:
        """
        :return: [us] time stamp in micro seconds
        """
        return self.time_point.time_us

    @property
    def time_seconds(self) -> float:
        """
        :return: [s] time stamp in seconds
        """
        return self.time_us * 1e-6


class StateVector3D:
    """Representation of vector in 3d."""

    __slots__ = "_x", "_y", "_z", "_array"

    def __init__(self, x: float, y: float, z: float):
        """
        Create StateVector3D object
        :param x: float direction
        :param y: float direction
        :param z: float direction
        """
        self._x = x  # x-axis in the vector.
        self._y = y  # y-axis in the vector.
        self._z = z  # y-axis in the vector.

        self._array: npt.NDArray[np.float64] = np.array([self.x, self.y, self.z], dtype=np.float64)

    def __repr__(self) -> str:
        """
        :return: string containing representation of this class
        """
        return f'x: {self.x}, y: {self.y}, z: {self.z}'

    def __eq__(self, other: object) -> bool:
        """
        Compare other object with this class
        :param other: object
        :return: true if other state vector is the same as self
        """
        if not isinstance(other, StateVector3D):
            return NotImplemented
        return bool(np.array_equal(self.array, other.array))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y, z]
        """
        return self._array

    @array.setter
    def array(self, other: npt.NDArray[np.float64]) -> None:
        """Custom setter so that the object is not corrupted."""
        self._array = other
        self._x = other[0]
        self._y = other[1]
        self._z = other[2]

    @property
    def x(self) -> float:
        """
        :return: x float state
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._x = x
        self._array[0] = x

    @property
    def y(self) -> float:
        """
        :return: y float state
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._y = y
        self._array[1] = y

    @property
    def z(self) -> float:
        """
        :return: z float state
        """
        return self._z

    @y.setter
    def z(self, z: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._z = z
        self._array[2] = z

    def magnitude(self) -> float:
        """
        :return: magnitude of vector
        """
        return float(np.linalg.norm(np.array([self.x, self.y, self.z])))
