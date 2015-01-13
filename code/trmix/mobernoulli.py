from mixture import Mixture
from bernoulli import Bernoulli

class MoBernoulli(Mixture):
	def __init__(self, dim=1, num_components=2, alpha=1., **kwargs):
		super(MoBernoulli, self).__init__(dim, alpha)

		for k in range(num_components):
			self.add_component(Bernoulli(dim=dim, **kwargs))
