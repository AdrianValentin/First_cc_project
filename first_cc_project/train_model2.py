import click
import torch
from torch import nn
from first_cc_project.models.model_day1 import myawesomemodel
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def processed_mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]

    train_data.append(torch.load(f"data/processed/processed_train_images.pt"))
    train_labels.append(torch.load(f"data/processed/train_targets.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("data/processed/processed_test_images.pt")
    test_labels = torch.load("data/processed/test_targets.pt")

    #print(train_data.shape)
    #print(train_labels.shape)
    #print(test_data.shape)
    #print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )


def train2(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    if lr >= 0.5:
        raise ValueError('Too high a learning rate')
    # TODO: Implement training loop here
    model = myawesomemodel.to(device)
    train_set, _ = processed_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    epoch_losses = []   
    for epoch in range(num_epochs):
        batch_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        print(f"Epoch {epoch} Loss {loss}")

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)

    torch.save(model, f"models/model{lr}_{batch_size}_{num_epochs}.pt")
 
     # Save the training curve
    plt.figure()
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    
    
    # Make sure the directory exists
    figures_directory = "reports/figures/"

    # Save the plot
    training_curve_path = f"{figures_directory}training_curve_lr{lr}_bs{batch_size}_epochs{num_epochs}.png"
    plt.savefig(training_curve_path)
    print(f"Training curve saved to {training_curve_path}")