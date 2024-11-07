import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.style.use('dark_background')

# Generate dataset
X, y = make_moons(5500, noise=0.15, random_state=42)

# Split dataset
X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_X = torch.from_numpy(X_train).to(torch.float32)
train_y = torch.from_numpy(y_train).to(torch.float32)
val_X = torch.from_numpy(x_val).to(torch.float32)
val_y = torch.from_numpy(y_val).to(torch.float32)

# Set batch size and create datasets
BATCH_SIZE = 64
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
train_X = train_X.to(DEVICE)
train_y = train_y.to(DEVICE)
val_X = val_X.to(DEVICE)
val_y = val_y.to(DEVICE)


# Define the model
class LinearRegression(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn((input_features, output_features), requires_grad=True))
        self.bias = nn.Parameter(torch.ones(output_features, requires_grad=True))

    def forward(self, x):
        x = x @ self.W
        x += self.bias
        return x


# Model creation and training cycle
###########################################################################################################################################################

# 1. Initializing model
model = LinearRegression(input_features=X.shape[1], output_features=1)
# 2. Select the loss function and set it
loss_function = nn.BCEWithLogitsLoss()
# 3. Set the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.042)
# 4. Moving all to the GPU
model = model.to(DEVICE)
accuracy = Accuracy(task='binary').to(DEVICE)

# Create auxiliary lists
losses_train = []
total_loss = []
accuracy_val = []
# Set the count epochs
MAX_EPOCHS = 100

for i in tqdm(range(MAX_EPOCHS)):
    model.train()
    epoch_losses = []  # Reset epoch losses for each epoch
    for X_batch, y_batch in train_loader:
        # Reset to zero gradients of optimizer
        optimizer.zero_grad()
        # Forward pass through the model
        outputs = model(X_batch.to(DEVICE)).view(-1)
        # Calculate loss
        loss = loss_function(outputs, y_batch.to(DEVICE))
        # Backpropagation
        loss.backward()
        # Track loss
        epoch_losses.append(loss.detach().cpu().numpy().item())
        # Next step of optimizer
        optimizer.step()
    total_loss.append(np.mean(epoch_losses))  # Store the mean loss of the epoch

    # Inference of model
    model.eval()
    with torch.no_grad():
        epoch_accuracy = []  # Reset epoch accuracy for validation
        for X_batch, y_batch in val_loader:
            # Forward pass for validation data
            outputs = model(X_batch.to(DEVICE)).view(-1)
            # Calculate loss
            loss = loss_function(outputs, y_batch.to(DEVICE))
            # Calculate probability
            proba = torch.sigmoid(outputs)
            # Count metric
            batch_acc = accuracy(proba, y_batch.to(torch.int32).to(DEVICE))
            epoch_accuracy.append(batch_acc.detach().cpu().numpy().item())
        accuracy_val.append(np.mean(epoch_accuracy))  # Store the mean accuracy of the epoch
