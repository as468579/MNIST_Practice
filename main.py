import argparse
from solver import Solver
from torchvision import datasets, transforms
import torch.utils.data as dset


def main(config):

	# Transform
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

	# Data
	trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
	testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)

	# Data loader
	trainLoader = dset.DataLoader(trainSet, batch_size=config.batch_size, shuffle=True)
	testLoader = dset.DataLoader(testSet, batch_size=config.batch_size, shuffle=False) 
	data_loader = [trainLoader, testLoader]

	# Solver for training and testing MNIST
	solver = Solver(data_loader, config)
	if config.mode == 'train':
		solver.train()
	else:
		solver.test()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Training configuration
	parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
	parser.add_argument('--num_iters', type=int, default=50, help='number of total iterations of training classifier')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate for classifier')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

	# Other
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

	config = parser.parse_args()
	main(config)