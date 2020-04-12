

from model import Classifier
from torch.optim import Adam
from datetime import datetime
import torch
import torch.nn as nn

class Solver():
	"""docstring for Solver"""
	def __init__(self, data_loader, config):

		self.trainLoader = data_loader[0]
		self.testLoader = data_loader[1]

		# Training configuration(????)
		self.num_iters = config.num_iters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.batch_size = config.batch_size

		self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
		self.build_model()

	def build_model(self):

		self.C = Classifier()
		self.c_optimizer = Adam(self.C.parameters(), self.lr, [self.beta1, self.beta2])

		self.print_network(self.C, 'Classifier')

		self.C.to(self.device)

	def print_network(self, model, name):
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters : {}".format(num_params))

	def cal_accuracy(self, x, labels):
		correct = 0
		total = 0
		with torch.no_grad():
			output = self.C(x)
			_, predicted = torch.max(output, 1)
			total = total + labels.size(0)
			correct = correct + (predicted == labels).sum().item()
		return 100 * correct / total

	def train(self):

		lr = self.lr
		start_iter = 0
		# norm = Normalizer()

		print("Start trainging......")
		start_time = datetime.now()

		for i in range(start_iter, self.num_iters):
			
			print("\n")
			print("Epochs : {}/{}".format(i+1, self.num_iters))
			print("=======================")
			loss_value = 0.0

			# Fetch images and labels
			for times, data in enumerate(self.trainLoader):
				# ================================================================== #
				#                            Get input data                          #
				# ================================================================== #

				x = data[0].reshape(data[0].shape[0], -1)
				labels = data[1]
				x = x.to(self.device)				# Input images
				labels = labels.to(self.device)		# Labels

				# ================================================================== #
				#                        Train the Classifier                        #
				# ================================================================== #
				
				# Zero the parameter gradients
				self.c_optimizer.zero_grad()

				# Compute loss with input images
				NLLLoss = nn.NLLLoss()
				output = self.C(x)
				loss = NLLLoss(output, labels)
				loss.backward()
				self.c_optimizer.step()

				loss_value += loss.item()

				# Print Loss values
				if times % 100 == 99 or times == len(self.trainLoader) - 1:
					print("times : {}/{}".format(times+1, len(self.trainLoader)))
					print("loss : {:.4f}".format(loss_value/times))
					print("acc : {:.4f}".format(self.cal_accuracy(x, labels)))
					print("=======================")




		print("Train Finish.")

if __name__ == "__main__":
	c = Classifier()
	print(c.parameters())