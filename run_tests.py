from constants import DEVICE
import torch

def train(NN, epoch_size, dataloader, optimizer, criterion):
	NN.train()
	for epoch in range(epoch_size):
		train_acc = 0
		i = 0
		train_running_loss = 0
		for X_batch, L_batch, batch_size, train_labels in dataloader:
			i += 1
			optimizer.zero_grad()
			prediction = NN(X_batch, L_batch, batch_size)
			loss = criterion(prediction, train_labels)
			loss.backward()
			optimizer.step()

			train_running_loss += loss.detach().item()
			train_acc += (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
			i += 1

		print(
			f"Epoch {epoch} | Train running loss {train_running_loss / i} | Loss {loss} | Train accuracy {train_acc / i}")


def test(NN, dataloader):
	NN.eval()

	test_acc = 0
	i = 0
	with torch.no_grad():
		for X_batch, L_batch, batch_size, test_labels in dataloader:
			prediction = NN(X_batch, L_batch, batch_size)
			test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
			i += 1

	print(f"Test accuracy of {test_acc / i}")