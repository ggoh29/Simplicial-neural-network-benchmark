from abc import abstractmethod, ABC
import torch.nn as nn

class GCNTemplate(nn.Module, ABC):

	def __init__(self, output_size):
		super().__init__()
		self.output_size = output_size


	@abstractmethod
	def forward(self, feature_dct):
		pass


	@abstractmethod
	def batch(self, scDataList):
		# Given a list of scData, batch it and store the fields in a feature_dct
		# returns feature_dct, label which is a dictionary, torch.tensor
		pass

	@abstractmethod
	def clean_feature_dct(self, feature_dct):
		# Torch sparse matrix cannot be used during multiprocessing. One way of getting past that is storing the
		# indices and values as separate tensors and combining them again when single threaded. This is done in this function
		pass