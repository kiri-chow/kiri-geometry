#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:01:04 2022

@author: kiri-chow

class to describe a geometric line

"""
import numpy as np
from kiri_geometry.regression import regress_plane
from .tools import (
    get_point, get_vector, normalize, ZERO,
    test_parallel, test_perpendicular, auto_ufunc_for_points, check_vector,
)


class BasePointVector:
    "base class to describe a geometric component with a point and a vector"

    # define the property: vector is a normal or direction
    _vector_is_normal = False

    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def __repr__(self):
        return f'{self.__class__.__name__}({self.point}, {self.vector})'

    @property
    def point(self):
        "a point on the instance"
        return self.__point

    @point.setter
    def point(self, point):
        self.__point = get_point(point)

    @property
    def vector(self):
        "the direction vector of the instance"
        return self.__vector

    @vector.setter
    def vector(self, vector):
        self.__vector = normalize(get_vector(vector))

    @property
    def _init_attributes(self):
        return (self.point, self.vector)

    def __and__(self, other):
        "return the cross point of self and other"
        # TODO : finish the code of method &

    def __floordiv__(self, other):
        "test if self and the other are parallel"
        parallel = test_parallel(self.vector, other.vector)
        if self._vector_is_normal + other._vector_is_normal == 1:
            parallel = not parallel
        return parallel

    def get_squared_distance(self, other):
        "return the squared distance bewteen self and the other"
        if isinstance(other, BasePointVector):
            return self._get_squared_distances_to_other(other)
        return self._get_squared_distances_to_points(other)

    def _get_squared_distances_to_other(self, other):
        # TODO : finish the code of method to _get_squared_distances_to_other
        pass

    def _get_squared_distances_to_points(self, points):
        "return the squared distance from self to each element in the points"
        return _get_squared_distances_to_points(points, self)

    def on(self, other, threshold=ZERO):
        """
        test if the other on self or self on the other

        params
        ------
        other : BasePointVector or np.ndarray
            an other BasePointVector object, or a array of points
        threshold : float,
            the maximum distance bewteen 2 close objects

        returns
        -------
        result : boolean or a array of boolean

        """
        if isinstance(other, BasePointVector):
            return self._test_other_on_self(other, threshold)
        return self._test_points_on_self(other, threshold)

    def _test_points_on_self(self, points, threshold=ZERO):
        "test if the points on self and return a array of boolean"
        squared_distances = _get_squared_distances_to_points(points, self)
        return squared_distances <= threshold

    def _test_other_on_self(self, other, threshold=ZERO):
        "test if the other on self and return a boolean"
        if self._vector_is_normal + other._vector_is_normal == 1:
            other, base = sorted(
                (self, other), key=lambda x: x._vector_is_normal)
            return (base._test_points_on_self(other.point, threshold)
                    and test_perpendicular(base.vector, other.vector))
        return (self._test_points_on_self(other.point, threshold)
                and test_parallel(self.vector, other.vector))

    def project(self, other, **kwargs):
        "project the other onto self"
        if isinstance(other, BasePointVector):
            return self._project_other(other)
        if check_vector(other):
            return self._convert_vector(other, **kwargs)
        return _project_points_to_point_vector(other, self)

    def _project_other(self, other):
        different_vector = self._vector_is_normal != other._vector_is_normal
        attrs = other._init_attributes
        attrs = [self.project(x, different_vector=different_vector)
                 for x in attrs]
        return type(other)(*attrs)

    def _convert_vector(self, vector, different_vector=False):
        if different_vector:
            return _get_perpendicular_vector(self.vector, vector)
        vector_new = self.vector.copy()
        if np.dot(vector, vector_new) < 0:
            vector_new *= -1
        return vector_new


class Line(BasePointVector):
    "class to describe a geometric line"

    _vector_is_normal = False


class Plane(BasePointVector):
    "class to describe a geometric plane"

    _vector_is_normal = True


class Segment(Line):
    "class to describe a geometric segment"

    _vector_is_normal = False

    def __init__(self, point_1, point_2):
        self.points = point_1, point_2

    @property
    def _init_attributes(self):
        return (self.points, )

    @property
    def points(self):
        "the starting point and the ending point of the segment"
        return self.__points

    @points.setter
    def points(self, points):
        point_1, point_2 = map(get_point, points)
        vector = point_2 - point_1
        self.__length = np.sqrt((vector[:3] ** 2).sum())
        self.__points = point_1, point_2

        self.point = point_1
        self.vector = vector

    @property
    def length(self):
        "the length of the segment"
        return self.__length


class Region(Plane):
    "class to describe a geometric region"

    def __init__(self, points):
        self.points = points

    @property
    def points(self):
        "the starting point and the ending point of the region"
        return self.__points

    @points.setter
    def points(self, points):
        points = np.asarry(points)
        if np.all(points[0] == points[-1]):
            points = points[:-1]
        assert len(points) >= 3
        point, vector, error = regress_plane(points[:, :3])
        self.point = point
        self.vector = vector
        self.__error = error
        self.__points = self.project(points)
        self.__segments = _get_segments_from_points(self.__points)

    @property
    def segments(self):
        "the segments of the region"
        return self.__segments

    @property
    def error(self):
        "the standard error of the regression"
        return self.__error


@auto_ufunc_for_points
def _get_squared_distances_to_points(points, point_vector):
    "return the squared distance from self to each element in the points"
    vectors = points - point_vector.point
    squared_dots = np.dot(point_vector.vector[:3], vectors.T[:3]) ** 2
    if point_vector._vector_is_normal:
        return squared_dots
    squared_lengths = (vectors ** 2).sum(1)
    squared_distances = squared_lengths - squared_dots
    return squared_distances


@auto_ufunc_for_points
def _project_points_to_point_vector(points, point_vector):
    "return the projected points on the point_vector"
    if point_vector._vector_is_normal:
        return _project_points_to_plane(points, point_vector)
    return _project_points_to_line(points, point_vector)


def _project_points_to_plane(points, point_vector):
    vectors = points - point_vector.point
    normal_vector = point_vector.vector
    dots = np.dot(normal_vector, vectors.T)
    return points - np.dot(normal_vector.reshape(4, 1),
                           dots.reshape(1, -1)).T


def _project_points_to_line(points, point_vector):
    vectors = points - point_vector.point
    one_vectors = (vectors.T / np.sqrt(np.sum(vectors ** 2, 1))).T

    normal_vectors = np.cross(
        list(map(normalize, np.cross(
            point_vector.vector[:3], one_vectors[:, :3]))),
        point_vector.vector[:3],
    )

    delta_vectors = np.array(list(map(
        lambda x: get_vector(x[0] * np.dot(x[0][:3], x[1][:3])),
        zip(normal_vectors, vectors))))

    return points - delta_vectors


def _get_perpendicular_vector(vector_base, vector):
    "convert vector to a new vector which perpendiculars to the vector_base"
    vector_new = normalize(np.cross(vector[:3], vector_base[:3]))
    vector_new = np.cross(vector_new, vector_base[:3])
    vector_new = get_vector(vector_new)
    if np.dot(vector, vector_new) < 0:
        vector_new *= -1
    return vector_new


def _cross_lines(line_1, line_2):
    "return the cross point of line_1 and line_2"
    if test_parallel(line_1.vector, line_2.vector):
        return None
    plane = Plane(line_1.point, np.cross(line_1.vector[:3], line_2.vector[:3]))
    if not plane.on(line_2.point):
        return None
    # TODO : finish this code, to calculate the cross point of 2 lines


def _get_segments_from_points(points):
    points = np.concatenate((points, points[:1]))
    segments = [Segment(*x) for x in zip(points[:-1], points[1:])]
    return segments
