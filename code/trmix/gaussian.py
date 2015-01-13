from copy import copy
from numpy import sum, zeros, eye, sqrt, arange, pi, log, dot
from numpy.linalg import cholesky, inv, slogdet
from numpy.random import randn
from scipy.special import psi

class Gaussian(object):
	def __init__(self, dim, m0=None, s0=None, psi0=None, nu0=None):
		self.dim = dim

		# normal-inverse-Wishart prior parameters
		self.m0 = zeros([dim, 1]) if m0 is None else m0
		self.s0 = 1. if s0 is None else s0
		self.psi0 = eye(dim) if psi0 is None else psi0
		self.nu0 = dim if nu0 is None else nu0

		if not self.nu0 > self.dim - 1:
			raise ValueError('nu0 must be greater than dim - 1.')

		# normal-inverse-Wishart beliefs
		L = cholesky(self.psi0 / (self.nu0 - self.dim + 1.) / 10.)
		self.m = copy(self.m0) + dot(L, randn(self.dim, 1))
		self.psi = copy(self.psi0)
		self.s = copy(self.s0)
		self.nu = copy(self.nu0)



	def update_parameters(self, data, phi, base, N, rho):
		# expected total number of documents assigned to this cluster
		N_k = sum(phi) * N / data.shape[1]

		# expected empirical mean
		x_bar = dot(data, phi.T / sum(phi))

		# expected empirical covariance structure
		x = (data - x_bar) * sqrt(phi)
		C = dot(x, x.T) * N / data.shape[1]

		self.s = (1. - rho) * base.s + rho * (self.s0 + N_k)
		self.nu = (1. - rho) * base.nu + rho * (self.nu0 + N_k)

		self.m = (1. - rho) * base.s * base.m + rho * (self.s0 * self.m0 + N_k * x_bar)
		self.m /= self.s

		xbm = x_bar - self.m

		self.psi = (1. - rho) * base.psi \
			+ rho * (self.psi0 + C + self.s0 * N_k / (self.s0 + N_k) * dot(xbm, xbm.T))



	def expected_log_likelihood(self, data):
		W = inv(self.psi)
		N = data.shape[1]
		x = data - self.m

		return -sum(x * dot(W, x), 0)[None, :] * self.nu / 2 \
			- self.dim / 2. / self.s \
			+ sum(psi((self.nu + 1. - arange(1, self.dim + 1)) / 2.)) / 2. \
			+ slogdet(W)[1] / 2. \
			- self.dim / 2. * log(pi)
