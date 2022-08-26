#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:01:46 2022

@author: kiri-chow

functions to initialize the geometric points and vectors

"""
from functools import wraps
import numpy as np

# define the zero value to aviod the error of the float-point arithmetic
ZERO = 1e-10


def get_point(point):
    "return an array formatted as a point"
    point = np.asarray(point)[:4]
    assert len(point.shape) == 1
    point = np.concatenate((point, np.zeros(4 - len(point))))
    point[-1] = 1
    return point


def get_vector(vector):
    "return an array formatted as a vector"
    vector = get_point(vector)
    vector[-1] = 0
    return vector


def check_vector(data):
    "test if the data is a vector"
    data = np.asarray(data)
    return data.shape == (4, ) and data[-1] == 0


def normalize(vector):
    "return a unit vector"
    length = np.sum(vector[:3] ** 2) ** 0.5
    vector[:3] /= length
    return vector


def test_parallel(vector_1, vector_2):
    "test if the vector_1 and vector_2 are parallel and return a boolean"
    return 1 - np.abs(np.dot(vector_1[:3], vector_2[:3])) <= ZERO


def test_perpendicular(vector_1, vector_2):
    "test if the vector_1 perpendiculars to the vector_2 and return a boolean"
    return np.abs(np.dot(vector_1[:3], vector_2[:3])) <= ZERO


def auto_ufunc_for_points(func):
    "decorator to process the @points in different shape"
    @wraps(func)
    def wrapper(points, *args, **kwargs):
        points = np.asarray(points)
        shape_length = len(points.shape)
        assert shape_length in {1, 2}
        if shape_length == 1:
            return func(points.reshape(1, -1), *args, **kwargs)[0]
        return func(points, *args, **kwargs)
    return wrapper
