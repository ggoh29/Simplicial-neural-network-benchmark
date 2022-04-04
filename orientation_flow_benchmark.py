import torch
from OrientationFlow.FlowDataset import FlowSCDataset
from models import flow_SAT, flow_ESNN, flow_BSNN, flow_SAN
from torch.utils.data import DataLoader
from constants import DEVICE


input_size = 1
output_size = 2
nb_epochs = 100
lr = 0.001
batch_size = 8
# f = torch.nn.functional.relu
f = torch.nn.Tanh()

# nn_mod = flow_SAT
nn_mod = flow_SAN
# nn_mod = flow_ESNN
# nn_mod = flow_BSNN

processor_type = nn_mod[0]
model = nn_mod[1]

model = model(input_size, input_size, input_size, output_size, f=f).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
loss_f = torch.nn.CrossEntropyLoss()

def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


if __name__ == "__main__":

    data = FlowSCDataset('./data', processor_type)
    train_dataset, test_dataset = data.get_val_train_split()

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                               shuffle=True, pin_memory=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8 ,
                             shuffle=True, pin_memory=True)

    for j in range(nb_epochs):
        training_acc, i = 0, 0
        model.train()
        for features_dct, train_labels in train_dataset:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = processor_type.repair(features_dct)
            features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
            train_labels = train_labels.to(DEVICE)
            optimiser.zero_grad()
            prediction = model(features_dct)
            loss = loss_f(prediction, train_labels)
            loss.backward()
            optimiser.step()

            train_acc = (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
            training_acc += (train_acc - training_acc) / i

        print(f"Training accuracy of {training_acc:.4f} for epoch {j}")

    model.eval()
    testing_acc = 0
    i = 0
    for features_dct, test_labels in test_dataset:
        features_dct = processor_type.clean_feature_dct(features_dct)
        features_dct = processor_type.repair(features_dct)
        features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
        test_labels = test_labels.to(DEVICE)
        prediction = model(features_dct)

        test_acc = (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
        i += 1
        testing_acc += (test_acc - testing_acc) / i

    print(f"Test accuracy of {testing_acc:.4f}")



