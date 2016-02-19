import theano
import theano.tensor as T

def brachitochrone_functional():
    # define all symbols
    lx, ly = T.dscalars('lx', 'ly')
    fseq = T.dvector('fseq')
    N = fseq.size
    delta_x = lx / N
    iseq = T.arange(N)

    # derivatives array
    dfseq, _ = theano.map(fn=lambda i: T.switch(T.eq(i, 0), fseq[1]-fseq[0], fseq[i]-fseq[i-1]) / delta_x,
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