import torch

def clean_batched_data(features, label):
    X_batch, I_batch, V_batch, batch_size = features
    # multiprocessing doesnt allow for sparse tensors to be storeed. So storing indices and values as separate tensors
    # and reconstructing the sparse matrix here.
    Lapacian_batch = []
    for i, v in zip(I_batch, V_batch):
        Lapacian_batch.append(torch.sparse_coo_tensor(i, v))
    return X_batch, Lapacian_batch, batch_size, label

def train(NN, epoch_size, dataloader, optimizer, criterion):
	NN.train()
	train_running_loss = 0
	for epoch in range(epoch_size):
		epoch_train_running_loss = 0
		train_acc = 0
		i = 0
		for features, train_labels in dataloader:
			X_batch, L_batch, batch_size, train_labels = clean_batched_data(features, train_labels)
			optimizer.zero_grad()
			prediction = NN(X_batch, L_batch, batch_size)
			loss = criterion(prediction, train_labels)
			loss.backward()
			optimizer.step()
			epoch_train_running_loss += loss.detach().item()
			train_acc += (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
			i += 1

		epoch_train_running_loss /= i
		train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
		print(
			f"Epoch {epoch} | Train running loss {train_running_loss} "
			f"| Loss {epoch_train_running_loss} | Train accuracy {train_acc / i}")


def test(NN, dataloader):
	NN.eval()

	test_acc = 0
	i = 0
	predictions = []
	with torch.no_grad():
		for features, test_labels in dataloader:
			X_batch, L_batch, batch_size, test_labels = clean_batched_data(features, test_labels)
			prediction = NN(X_batch, L_batch, batch_size)
			test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
			predictions.append(prediction)
			i += 1

	print(f"Test accuracy of {test_acc / i}")
	return predictions