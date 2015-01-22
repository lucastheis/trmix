import sys

sys.path.append('./code')

from glob import glob
from argparse import ArgumentParser
from numpy import hstack, asarray, dot, cov, load, savez
from numpy.linalg import cholesky, inv
from numpy.random import rand, randn, randint, permutation, multinomial, dirichlet, seed
from trmix import MoGaussian
from matplotlib.pyplot import *

def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--data_K', '-K', type=int,   default=10,
		help='Number of components used for generating data.')
	parser.add_argument('--data_D', '-D', type=int,   default=2,
		help='Dimensionality of data.')
	parser.add_argument('--data_a', '-a', type=float, default=.8,
		help='Concentration parameter used for generating data.')
	parser.add_argument('--data_N', '-N', type=int,   default=1000,
		help='Number of generated data points.')
	parser.add_argument('--max_iter_tr', '-m', type=int, default=5,
		help='Number of steps in the inner loop of the trust-region method.')
	parser.add_argument('--batch_size', '-b', type=int, default=100)
	parser.add_argument('--seed', '-s', type=int, default=1,
		help='Random seed for generating data.')
	parser.add_argument('--max_epochs', '-E', type=int, default=500)

	args = parser.parse_args(argv[1:])

	# matplotlib
	ion()

	# load results of previous runs
	results = []
	for filepath in glob('results/mog/lb.*.npz'):
		results.append(load(filepath))
	
	# data parameters
	K = args.data_K
	D = args.data_D
	a = args.data_a
	N = args.data_N 

	# generate data
	sd = randint(2**32 - 1)
	seed(args.seed)

	p = dirichlet([a] * K)
	n = multinomial(N, p)
	m = []
	C = []
	X = []

	for k in range(K):
		m.append(rand(D, 1) * 20. - 10.)
		C.append(inv(cov(randn(D, D * D))) / 5.)
		X.append(dot(cholesky(C[k]), randn(D, n[k])) + m[k]) 

	data = hstack(X)[:, permutation(N)]

	# create model

	model = MoGaussian(D, K * 5, alpha=2. / K, s0=.1)

	def callback(model):
		# count number of updates
		callback.counter += 1

		if callback.counter * args.max_iter_tr % (40 * max(1, args.batch_size // 100)) == 0:
			callback.num_updates.append(callback.counter * args.max_iter_tr)
			callback.lower_bound.append(model.lower_bound(data).ravel()[0])

			clf()
			for result in results:
				plot(result['num_updates'], result['lower_bound'], color=(.5, .5, .5), lw=1)
			plot(callback.num_updates, callback.lower_bound, 'k', lw=2)
			xlim([0, (args.max_epochs + 50) * (args.data_N // args.batch_size)])
#			ylim([-6000, -4000])
			draw()

	callback.counter = 0
	callback.lower_bound = []
	callback.num_updates = []

	try: 
		# initialize model
#		model.train(datadrop, 
#			batch_size=args.batch_size, 
#			rho=.05,
#			max_iter_tr=args.max_iter_tr,
#			max_epochs=50 // args.max_iter_tr,
#			callback=callback,
#			init='uniform',
#			update_gamma=False)

		# train model
		model.train(data, 
			batch_size=data.shape[1],# args.batch_size, 
			rho=1.,
			max_iter_tr=args.max_iter_tr,
			max_epochs=args.max_epochs // args.max_iter_tr,
			callback=callback,
			init='uniform',
			update_gamma=True)

	except KeyboardInterrupt:
		pass

	savez('results/mog/lb.{0}.npz'.format(len(results)),
		num_updates=callback.num_updates,
		lower_bound=callback.lower_bound)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
