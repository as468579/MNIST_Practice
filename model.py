import torch
import torch.nn as nn
from torchvision import datasets, transforms

class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = 10
		self.main = nn.Sequential(nn.Linear(784, 128), 
								  nn.ReLU(),
								  nn.Linear(128, 64),
								  nn.ReLU(),
								  nn.Linear(64, self.num_classes),
								  nn.LogSoftmax())

	def forward(self, x):
		return self.main(x)

# if __name__ == '__main__':


	# t = torch.rand([28, 28, 1])
	# t = t.view(28*28)
	# C = Classifier()
	# o = C(t)
	# print(o.shape)