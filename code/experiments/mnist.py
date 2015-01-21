"""
Train a mixture of Bernoullis on MNIST data.
"""

import sys

sys.path.append('./code')

from argparse import ArgumentParser
from matplotlib.pyplot import imsave
from tools import stitch
from numpy import load, round, asarray, hstack
from numpy.random import rand, permutation
from trmix import MoBernoulli

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--num_components', '-c', type=int, default=20)
	parser.add_argument('--max_epochs', '-e', type=int, default=4)

	args = parser.parse_args(argv[1:])

	data = load('data/mnist.npz')['train']
	data = data[:, permutation(data.shape[1])]
	data = asarray(data, dtype=float) / 255.
	data = asarray(rand(*data.shape) < data, dtype=float, order='F')

	def callback(model):
		if model.num_updates % 5:
			return

		print model.lower_bound(data)

		p = []
		for k in range(len(model)):
			p.append(model[k].alpha / (model[k].alpha + model[k].beta))
		p = hstack(p)

		imsave('results/mnist/{0}.png'.format(model.num_updates // 2),
			stitch(p.T.reshape(-1, 28, 28), num_rows=4), cmap='gray', vmin=0., vmax=1.)

	model = MoBernoulli(dim=784, num_components=args.num_components)
	model.train(data,
		batch_size=200,
		max_epochs=args.max_epochs,
		tau=100.,
		callback=callback)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
