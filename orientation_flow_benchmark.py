import torch
from OrientationFlow.FlowDataset import FlowSCDataset
from models import flow_SAT, flow_SCN, flow_SCConv, flow_SAN
from torch.utils.data import DataLoader
from constants import DEVICE


input_size = 1
output_size = 2
nb_epochs = 100
lr = 0.001
batch_size = 4

# f = torch.nn.functional.relu
f = torch.nn.Tanh()
# f = torch.nn.Identity()

# nn_mod = flow_SAT
# nn_mod = flow_SAN
nn_mod = flow_SCN
# nn_mod = flow_SCConv

processor_type = nn_mod[0]
model = nn_mod[1]

model = model(input_size, input_size, input_size, output_size, f=f).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
loss_f = torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    data = FlowSCDataset('./data', processor_type)
    train_dataset, test_dataset = data.get_val_train_split()

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                               shuffle=True, pin_memory=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                             shuffle=True, pin_memory=True)

    best_acc = 0
    for j in range(nb_epochs):
        training_acc, i = 0, 0
        model.train()
        for simplicialComplex, train_labels in train_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            simplicialComplex.to_device()
            train_labels = train_labels.to(DEVICE)
            optimiser.zero_grad()
            prediction = model(simplicialComplex)
            loss = loss_f(prediction, train_labels)
            loss.backward()
            optimiser.step()

            train_acc = (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
            training_acc += (train_acc - training_acc) / i
        if training_acc > best_acc:
            torch.save(model.state_dict(), f'./data/{model.__class__.__name__}_flow.pkl')
            best_acc = training_acc

        print(f"Training accuracy of {training_acc:.4f} for epoch {j}")

    model.load_state_dict(torch.load(f'./data/{model.__class__.__name__}_flow.pkl'))
    model.eval()
    testing_acc = 0
    i = 0
    with torch.no_grad():
        for simplicialComplex, test_labels in test_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            simplicialComplex.to_device()
            test_labels = test_labels.to(DEVICE)
            prediction = model(simplicialComplex)

            test_acc = (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
            i += 1
            testing_acc += (test_acc - testing_acc) / i

    print(f"Test accuracy of {testing_acc:.4f}")



