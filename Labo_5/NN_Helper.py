__author__ = 'odjansse'

import numpy as np

class Gradient_Checker:


    def rel_error(self,x, y):
          return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    def __init__(self,limit=1.0*np.exp(-8)):
        self.limit = limit

    def gradient_check(self,X,y,dinputdoutput,f):

        grad = self.eval_numerical_gradient(f,X)
        error = self.rel_error(dinputdoutput,grad)

        if error <= self.limit:
            print "Good gradient, difference is: "+str(error)
        else:
            print "Bad gradient, difference is: "+ str(error)


    def eval_numerical_gradient(self,f, x, verbose=False, h=0.00001):
      """
      a naive implementation of numerical gradient of f at x
      - f should be a function that takes a single argument
      - x is the point (numpy array) to evaluate the gradient at
      """

      fx = f(x) # evaluate function value at original point
      grad = np.zeros_like(x)
      # iterate over all indexes in x
      it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
      while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
          print ix, grad[ix]
        it.iternext() # step to next dimension

      return grad