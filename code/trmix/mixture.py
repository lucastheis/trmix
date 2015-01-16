from numpy import asarray, vstack, sum, mean, empty, exp, ones, zeros
from numpy.random import rand, permutation
from scipy.special import psi
from utils import logsumexp
from copy import copy, deepcopy
from time import sleep
from six import string_types

class Mixture(object):
	def __init__(self, dim=1, alpha=1.):
		self.dim = dim
		self.components = []

		 # Dirichlet prior over mixture weights
		self.alpha = alpha

		# Dirichlet beliefs over mixture weights
		self.gamma = None

		self.num_updates = 0



	def __len__(self):
		return len(self.components)



	def __getitem__(self, key):
		return self.components[key]



	def add_component(self, component):
		if component.dim != self.dim:
			raise ValueError('Mixture component has wrong dimensionality.')

		self.gamma = asarray([[self.alpha]]) if not self.components \
			else vstack([self.gamma, self.alpha])
		self.components.append(component)



	def posterior(self, data):
		"""
		Compute posterior over component indices.
		"""

		log_post = empty([len(self), data.shape[1]])

		for k in range(len(self.components)):
			log_post[k] = self.components[k].expected_log_likelihood(data)

		log_post += (psi(self.gamma) - psi(sum(self.gamma)))
		log_post -= logsumexp(log_post, 0)

		return exp(log_post)



	def update_parameters(self, data, kappa=.5, tau=10., max_iter_tr=10, rho=None, N=None,
			init='none', update_gamma=True):
		"""
		Perform one trust-region update.

		@type  init: tuple/str 
		@param init: initialization strategy (one of 'drop', 'uniform', 'none')
		"""

		if N is None:
			# assume we're given all the data
			N = data.shape[1]

		if isinstance(init, string_types):
			init = (init, {})

		# learning rate
		self.num_updates += 1
		if rho is None:
			rho = pow(tau + self.num_updates, -kappa)

		# copy current parameters
		components = deepcopy(self.components)
		gamma = copy(self.gamma)

		for i in range(max_iter_tr):
			# E-step
			phi = self.posterior(data)
			
			# apply heuristic for initialization
			if max_iter_tr > 1 and i == 0:
				if init[0].lower() == 'drop':
					phi *= rand(*phi.shape) < init[1].get('p', .5)
				elif init[0].lower() == 'normalize':
					phi += 1e-10
					phi /= sum(phi, 1)[:, None] * init[1].get('K', N / float(len(self)))
				elif init[0].lower() == 'uniform':
					phi = ones(phi.shape) / len(self)
				elif init[0].lower() == 'shuffle':
					phi = phi[permutation(len(self))]
				elif init[0].lower() != 'none':
					raise ValueError('Unknown initialization strategy \'{0}\'.'.format(
						init[0]))

			# M-step
			if update_gamma:
				self.gamma = (1 - rho) * gamma + rho * (self.alpha + N * mean(phi, 1)[:, None])

			for k in range(len(self)):
				self.components[k].update_parameters(
					data, phi=phi[[k]], base=components[k], N=N, rho=rho)



	def train(self, data, batch_size=50, max_epochs=10, callback=None, **kwargs):
		kwargs['N'] = data.shape[1]

		if callback:
			callback(self)

		for epoch in range(max_epochs):
			for b in range(0, data.shape[1], batch_size):
				self.update_parameters(data[:, b:b + batch_size], **kwargs)
				if callback:
					callback(self)
