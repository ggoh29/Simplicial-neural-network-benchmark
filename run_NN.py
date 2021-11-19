import time

import torch
from constants import DEVICE

def convert_to_device(lst):
	return [i.to(DEVICE) for i in lst]

def train(NN, epoch_size, dataloader, optimizer, criterion, processor_type):
	NN.train()
	train_running_loss = 0
	t = 0
	for epoch in range(epoch_size):
		t1 = time.perf_counter()
		epoch_train_running_loss = 0
		train_acc = 0
		i = 0
		for features_dct, train_labels in dataloader:
			features_dct = processor_type.clean_feature_dct(features_dct)
			features_dct = {key : convert_to_device(features_dct[key]) for key in features_dct}
			train_labels = train_labels.to(DEVICE)
			optimizer.zero_grad()
			prediction = NN(features_dct)
			loss = criterion(prediction, train_labels)
			loss.backward()
			optimizer.step()
			epoch_train_running_loss += loss.detach().item()
			train_acc += (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
			i += 1
		t2 = time.perf_counter()
		t = (t * epoch + (t2 - t1)) / (epoch + 1)
		epoch_train_running_loss /= i
		train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
		print(
			f"Epoch {epoch} | Train running loss {train_running_loss} "
			f"| Loss {epoch_train_running_loss} | Train accuracy {train_acc / i}")
		epoch_loss = epoch_train_running_loss
		acc = train_acc / i
	return t, train_running_loss, epoch_loss, acc


def test(NN, dataloader, processor_type):
	NN.eval()

	test_acc = 0
	i = 0
	predictions = []
	with torch.no_grad():
		for features_dct, test_labels in dataloader:
			features_dct = processor_type.clean_feature_dct(features_dct)
			features_dct = {key : convert_to_device(features_dct[key]) for key in features_dct}
			test_labels = test_labels.to(DEVICE)
			prediction = NN(features_dct)
			test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
			predictions.append(prediction)
			i += 1

	print(f"Test accuracy of {test_acc / i}")
	return predictions, (test_acc / i)