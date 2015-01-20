"""
KL divergence of normal-inverse-Wishart distribution.
"""

import theano as th
import theano.tensor as tt
import theano.sandbox.linalg as tl
import numpy as np

def log_partf(b, s, C, v):
    D = b.size
    
    # multivariate log-gamma function
    g = tt.sum(tt.gammaln((v + 1. - tt.arange(1, D + 1)) / 2.)) + D * (D - 1) / 4. * np.log(np.pi)

    # log-partition function
    return -v / 2. * tt.log(tl.det(C - tt.dot(b, b.T) / (4 * s))) \
        + v * np.log(2.) + g - D / 2. * tt.log(s)

# natural parameters
b1 = tt.dmatrix('b1')
s1 = tt.dscalar('s1')
C1 = tt.dmatrix('C1')
v1 = tt.dscalar('v1')

b2 = tt.dmatrix('b2')
s2 = tt.dscalar('s2')
C2 = tt.dmatrix('C2')
v2 = tt.dscalar('v2')

# log-partition functions
A1 = log_partf(b1, s1, C1, v1)
A2 = log_partf(b2, s2, C2, v2)

# KL divergence
D_KL = A2 - A1 \
    + th.dot((b1 - b2).T, tt.grad(A1, b1)) \
    + tl.trace(th.dot(C1 - C2, tt.grad(A1, C1))) \
    + (s1 - s2) * tt.grad(A1, s1) \
    + (v1 - v2) * tt.grad(A1, v1)
    
KL_divergence = th.function([b1, s1, C1, v1, b2, s2, C2, v2], D_KL)
