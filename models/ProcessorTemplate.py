from abc import abstractmethod, ABC
from Superpixel.SuperpixelDataset.ImageProcessor import SCData


class NNProcessor(ABC):

	@abstractmethod
	def process(self, scData: SCData):
		# Given a scData object, continue to process it until the structure can be stored in inmemorydataset
		pass

	@abstractmethod
	def collate(self, objectList: list):
		# Given a list of objects which we have chosen to represent out dataset, combine into one big object to write to memory
		pass

	@abstractmethod
	def get(self, data, slice : dict, idx : int):
		# Given an index and a collated object, take out the individual object
		pass

	@abstractmethod
	def batch(self, objectList: list):
		# Given a list of objects which are representations how we want to store data for each model, batch it and
		# store the fields in a feature_dct.
		# returns feature_dct, label which is a dictionary, torch.tensor
		pass

	@abstractmethod
	def clean_feature_dct(self, feature_dct: dict):
		# Torch sparse matrix cannot be used during multiprocessing. One way of getting past that is storing the
		# indices and values as separate tensors and combining them again when single threaded. This is done in this function
		pass

