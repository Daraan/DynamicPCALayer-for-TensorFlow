# -*- coding: utf-8 -*-
"""
Based on rafaelvalle work from 
https://stackoverflow.com/a/37121355

Calculates the elbow value by finding the data point
with the maximum distance to the line through
the first and last data value.
"""

import numpy as np
from Typing import List

def elbow_index(Ydata : List, Xdata : List=None):
    """
    Returns the index of the elbow value in the given data.
    The calculate_elbow method is preferred.
    Without Xdata this assumes a range(0, len(Ydata)) as Xdata.
    """
    XY = np.empty((len(Ydata),2))
    XY[:,0] = Xdata or range(len(Ydata))
    XY[:,1] = Ydata
    dir_vec = XY[-1] - XY[0]
    dir_vec = dir_vec / np.sqrt(np.sum(dir_vec**2)) # norm
    XY = XY - XY[0]       # set to origin
    
    scalarProduct = XY @ dir_vec
    
    XYParallel = np.outer(scalarProduct, dir_vec) # outer product to all points
    vecToLine = XY - XYParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    return np.argmax(distToLine)


def calculate_elbow(Ydata : List, Xdata : List):
    """
    Returns the X value where the point of maximum distance
    the elbow value is of the Ydata.
    """
    return elbow_index(Ydata, Xdata)


