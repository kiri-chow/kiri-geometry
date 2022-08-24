#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:46:27 2022

@author: kiri-chow

"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


class RegressionError(BaseException):
    "error raised in the regression"


def regress_line(data):
    '''
    regress a line by giving data

    Parameters
    ----------
    data : array-like
        the points to regress

    Raises
    ------
    RegressionError

    Returns
    -------
    point : np.ndarray,
        a point on the regression line
    vector : np.ndarray,
        the direction vector of the regression line
    std: np.float,
        standard error of the regression

    '''
    # check & init data
    data = _init_data(data)

    # regression
    point = np.mean(data, 0)
    vectors = data - point
    vector = PCA(1).fit(vectors).components_[0]

    # calculate std
    std = _calculate_line_standard_error(data, point, vector)
    
    return point, vector, std



def regress_line_by_neighbors(data, radius=50, min_points=3, max_std=1, **kwargs):
    """
    regress a line by regressing all points' neighborhood

    Parameters
    ----------
    data : array-like
        the points to regress
    radius : float
        distance within wihch neighbors are included
    min_samples : int,
        the minimum number of samples in a neighborhood to regress
    max_std : float,
        the maximum standard error of a valid regression

    kwargs
    ------
    other init parameters of sklearn.neighbors.KDTree

    Raises
    ------
    RegressionError

    Returns
    -------
    point : np.ndarray,
        a point on the regression line
    vector : np.ndarray,
        the direction vector of the regression line
    std: np.float,
        standard error of the regression

    """
    # check & init the data
    data = _init_data(data)

    # regress each neighborhood
    tree = KDTree(data, **kwargs)
    data_to_reg = tree.query_radius(data, radius)
    result = filter(
        bool,
        map(lambda x: _base_reg_nebr(x, data, min_points, max_std),
            data_to_reg,
            ))
    try:
        points, vectors = zip(*result)
    except ValueError:
        raise RegressionError("dont have enough points to regress")

    # build data of the line
    point = np.mean(points, 0)
    vectors = np.array(vectors)
    vectors[vectors[0].dot(vectors.T) < 0] *= -1
    vector = np.sum(vectors, 0)
    vector /= sum(vector ** 2) ** 0.5

    # calculate the standard error
    std = _calculate_line_standard_error(data, point, vector)

    return point, vector, std


def _init_data(data):
    if len(data) == 0:
        raise RegressionError('the length of data should not be 0')
    data = np.asarray(data)
    return data


def _base_reg_nebr(index, data, min_points, max_std):
    if len(index) < min_points:
        return None
    point, vector, std = regress_line(data[index])
    if std > max_std:
        return None
    return point, vector


def _calculate_line_standard_error(data, point, vector):
    vectors_data = data - point
    distances_data = np.sqrt(
        np.sum(vectors_data ** 2, 1) - np.dot(vector, vectors_data.T) ** 2)
    std = np.std(distances_data)
    return std


def reg_plane(data):
    '''
    regress a plane by giving data

    Parameters
    ----------
    data : array-like
        the points to regress

    Raises
    ------
    RegressionError

    Returns
    -------
    point : np.ndarray,
        a point on the regression plane
    vector : np.ndarray,
        the normal vector of the regression plane
    std: np.float,
        standard error of the regression

    '''
    data = _init_data(data)

    point = np.mean(data, 0)
    vectors = data - point
    VT = np.linalg.svd(vectors, full_matrices=False)[-1]
    vector = VT.T[:, -1]

    # ensure the vector aims to the zero point
    if np.dot(vector, point) >= 0:
        vector = -vector

    std = _calculate_plane_standard_error(data, point, vector)

    return point, vector, std


def _calculate_plane_standard_error(data, point, vector):
    vectors_data = data - point
    distances_data = np.abs(np.dot(vector, vectors_data.T))
    std = np.std(distances_data)
    return std

