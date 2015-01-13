from numpy import ones, dot, sum
from numpy.random import gamma
from scipy.special import psi

class Bernoulli(object):
	def __init__(self, dim=1, a=1., b=1.):
		self.dim = dim

		# beta prior parameters
		self.a = a
		self.b = b

		# beta beliefs
		self.alpha = gamma(100, 0.01, [dim, 1])
		self.beta = gamma(100, 0.01, [dim, 1])



	def update_parameters(self, data, phi, base, N, rho):
		self.alpha = (1. - rho) * base.alpha \
			+ rho * (self.a + N * dot(data, phi.T) / data.shape[1])
		self.beta = (1. - rho) * base.beta \
			+ rho * (self.a + N * dot((1 - data), phi.T) / data.shape[1])



	def expected_log_likelihood(self, data):
		return dot(psi(self.alpha).T, data) \
			+ dot(psi(self.beta).T, 1 - data) \
			- sum(psi(self.alpha + self.beta))


	def p(self):
		return self.alpha / (self.alpha + self.beta)
