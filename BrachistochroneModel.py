import theano
import theano.tensor as T
import numpy as np
from operator import and_

def brachistochrone_functional():
    # define all symbols
    lx, ly = T.dscalars('lx', 'ly')
    fseq = T.dvector('fseq')
    N = fseq.size
    delta_x = lx / N
    iseq = T.arange(N)

    # derivatives array
    dfseq, _ = theano.map(fn=lambda i: T.switch(T.eq(i, 0), fseq[0]-ly, fseq[i]-fseq[i-1]) / delta_x,
                          sequences=[iseq])

    # functional term
    functional_ithterm = lambda i: delta_x * T.sqrt(0.5*(1+(dfseq[i])**2)/(ly-fseq[i]))

    # defining the functions
    functional_parts, _ = theano.map(fn=lambda k: functional_ithterm(k), sequences=[iseq])
    functional = functional_parts.sum()
    gfunc = T.grad(functional, fseq)

    # compile the functions
    time_fcn = theano.function(inputs=[fseq, lx, ly], outputs=functional)
    grad_time_fcn = theano.function(inputs=[fseq, lx, ly], outputs=gfunc)

    return time_fcn, grad_time_fcn

def brachistochrone_gradient_descent(lx, ly, N, learn_rate=0.1, tol=1e-7, max_iter=10000):
    # get compiled function
    time_fcn, grad_time_fcn = brachistochrone_functional()

    # initialize points
    x = np.linspace(0, lx, N+1)[1:]
    # get a linear line for y first
    y = np.linspace(ly, 0, N+1)[1:]

    # loop
    step = 0
    converged = False
    while step < max_iter and not converged:
        new_y = y - learn_rate * grad_time_fcn(y, lx, ly)
        converged = reduce(and_, np.abs(new_y-y)<tol)
        y = new_y
        step += 1

    return x, y