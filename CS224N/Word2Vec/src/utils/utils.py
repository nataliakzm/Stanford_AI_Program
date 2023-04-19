#!/usr/bin/env python

import json
import numpy as np

def dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def load(path):
    with open(path) as f:
        obj = json.load(f)
    return obj

def normalize_rows(x):
    """ Row normalization function
    Implement a function that normalizes each row of a matrix to have unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape((N, 1)) + 1e-30
    return x


def softmax(x):
    """Compute the softmax function for each row of the input x.
    Arguments: x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return: x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:

        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x
