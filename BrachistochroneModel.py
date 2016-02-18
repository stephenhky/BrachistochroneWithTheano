import theano
import theano.tensor as T

def brachitochrone_functional():
    lx, ly = T.dscalars('lx', 'ly')
    N = T.iscalar('N')
    delta_x = lx / N
    iseq = T.arange(N)
    fseq = T.dvector('fseq')

    functional_1stterm = delta_x * T.sqrt(0.5*(1+((fseq[0]-ly)/delta_x)**2)/(ly-fseq[0]))
    functional_lastterm = delta_x * T.sqrt(0.5*(1+((fseq[N-1]-fseq[N-2])/delta_x)**2)/(ly-fseq[N-1]))
    functional_ithterm = lambda i: delta_x * T.sqrt(0.5*(1+((fseq[i+1]-fseq[i-1])/(2*delta_x))**2)/(ly-fseq[i]))

    pfunc = lambda i: T.switch(T.eq(i, 0),
                               functional_1stterm,
                               T.switch(T.eq(i, N-1),
                                        functional_lastterm,
                                        functional_ithterm(i)))

    functional_parts = theano.map(fn=lambda k: pfunc(k), sequences=[iseq])
    functional = functional_parts.sum()
    gfunc = T.grad(functional, fseq)

    time = theano.function(inputs=[fseq, lx, ly, N], outputs=functional)
    grad_time = theano.function(inputs=[fseq, lx, ly, N], outputs=gfunc)

    return time, grad_time