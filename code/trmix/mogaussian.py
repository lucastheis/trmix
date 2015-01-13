from mixture import Mixture
from gaussian import Gaussian

class MoGaussian(Mixture):
	def __init__(self, dim=1, num_components=2, alpha=1., **kwargs):
		super(MoGaussian, self).__init__(dim, alpha)

		for k in range(num_components):
			self.add_component(Gaussian(dim=dim, **kwargs))
