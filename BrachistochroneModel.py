import theano
import theano.tensor as T
import numpy as np
from operator import and_

def brachistochrone_functional():
    # define all symbols
    lx, ly = T.dscalars('lx', 'ly')
    fseq = T.dvector('fseq')
    N = fseq.size + 1
    delta_x = lx / N
    iseq = T.arange(N-1)

    # functional term
    functional_ithterm = lambda i: T.switch(T.eq(i, 0),
                                            T.sqrt(0.5*(delta_x**2+(fseq[0]-ly)**2)/(ly-0.5*(fseq[0]+ly))),
                                            T.sqrt(0.5*(delta_x**2+(fseq[i]-fseq[i-1])**2)/(ly-0.5*(fseq[i]+fseq[i-1])))
                                            )

    # defining the functions
    functional_parts, _ = theano.map(fn=lambda k: functional_ithterm(k), sequences=[iseq])
    functional = functional_parts.sum() + T.sqrt(0.5*(delta_x**2+(0-fseq[N-2])**2)/(ly-0.5*(0+fseq[N-2])))
    gfunc = T.grad(functional, fseq)

    # compile the functions
    time_fcn = theano.function(inputs=[fseq, lx, ly], outputs=functional)
    grad_time_fcn = theano.function(inputs=[fseq, lx, ly], outputs=gfunc)

    return time_fcn, grad_time_fcn

def brachistochrone_gradient_descent(lx, ly, N, learn_rate=0.001, tol=1e-7, max_iter=1e+7):
    # get compiled function
    time_fcn, grad_time_fcn = brachistochrone_functional()

    # initialize points
    x = np.linspace(0, lx, N+1)[1:-1]
    # get a linear line for y first
    y = np.linspace(ly, 0, N+1)[1:-1]

    # loop
    step = 0
    converged = False
    while step < max_iter and not converged:
        new_y = y - learn_rate * grad_time_fcn(y, lx, ly)
        converged = reduce(and_, np.abs(new_y-y)<tol)
        y = new_y
        step += 1

    return x, y, {'steps': step, 'converged': converged}