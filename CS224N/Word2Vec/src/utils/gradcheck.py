#!/usr/bin/env python

import random

import numpy as np


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x, gradient_text):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4 

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x[ix] += h  # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x)  # evalute f(x + h)
        x[ix] -= 2 * h  # restore to previous value (very important!)
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for %s." % gradient_text)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return False

        it.iternext()  # Step to next dimension

    print("Gradient check passed!")
    return True
