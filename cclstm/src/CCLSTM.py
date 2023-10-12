import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
import time
from sklearn.metrics import r2_score

def read_csv_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)  # Get the directory of your script
    file_path = os.path.join(script_directory, file_name)
    return pd.read_csv(file_path)

class CCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CCModel, self).__init__()
        # LSTM layers
        self.lstm_layers = []
        self.lstm_layers.append(nn.LSTMCell(input_size, hidden_size, batch_first=True))
        self.tot_size = hidden_size
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        # Output layer
        self.output_layer = nn.Linear(self.tot_size, output_size)

    def forward(self, x): # x is pytorch tensor
        out = []
        out.append(x.tolist())
        # For each layer, add all outputs to the output list, which is passed as the input into the next layer
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(torch.tensor(out, dtype=torch.float32))
            out.append(x.tolist())
        # Use only the last timestep's output
        x = self.output_layer(torch.tensor(out, dtype=torch.float32)[:, -1, :])
        return x
    
    def add_layer(self):
        self.lstm_layers.append(nn.LSTMCell(self.prev_size, 1, batch_first=True))
        self.tot_size = self.tot_size + 1

# Define a directory to save checkpoints
checkpoint_dir = 'checkpoints_new'
os.makedirs(checkpoint_dir, exist_ok=True)
# Define the frequency of saving checkpoints (e.g., every 10 epochs)
checkpoint_frequency = 20

# Choose which dataset to use (covid, stock, or wind)
datatype = input("Choose a dataset (covid, stock, or wind): ")
if datatype == "covid":
    input_size = 35
    hidden_size = 3
    learning_rate = 0.01
    num_epochs = 10
    target = 'new_deaths_per_million'
    loss_threshold = 0.1
    num_epochs_tot = 1000
elif datatype == "stock":
    input_size = ...
    hidden_size = ...
    learning_rate = ...
    num_epochs = ...
    target = ''
    loss_threshold = 0.1
    num_epochs_tot = 1000
elif datatype == "wind":
    input_size = ...
    hidden_size = ...
    learning_rate = ...
    num_epochs = ...
    target = ''
    loss_threshold = 0.1
    num_epochs_tot = 1000

# Load your dataset (replace 'datatype' with the appropriate dataset type)
dataset = pd.read_csv(datatype+'_data_fixed.csv')
# Define the sequence length (window size)
sequence_length = 10 
# Define the overlap percentage (50% overlap)
overlap_percentage = 0.5
# Calculate the overlap step based on the sequence length and overlap percentage
overlap_step = int(sequence_length * overlap_percentage)
# Create sequences for training, validation and testing using the sliding window method with overlap
sequences_train = []
sequences_val = []
sequences_test = []
target_train = []  # List to store target sequences for training
target_val = []    # List to store target sequences for validation
target_test = []   # List to store target sequences for testing
i = 0
while i + sequence_length <= len(dataset):
    # Extract input sequence
    input_sequence = dataset.iloc[i:i+sequence_length].values
    # Extract corresponding target for the next time step
    target_sequence = dataset.iloc[i+sequence_length][target]
    
    if i < len(dataset) * 0.8:
        sequences_train.append(input_sequence)
        target_train.append(target_sequence)
    elif i < len(dataset) * 0.9:
        sequences_val.append(input_sequence)
        target_val.append(target_sequence)
    else:
        sequences_test.append(input_sequence)
        target_test.append(target_sequence)
    i += overlap_step
# Convert input and target sequences to PyTorch tensors
train_input_dataset = torch.FloatTensor(sequences_train)
val_input_dataset = torch.FloatTensor(sequences_val)
test_input_dataset = torch.FloatTensor(sequences_test)
train_target_dataset = torch.FloatTensor(target_train)
val_target_dataset = torch.FloatTensor(target_val)
test_target_dataset = torch.FloatTensor(target_test)
# Define the training, validation and test input dataloaders
train_input_dataloader = DataLoader(train_input_dataset, batch_size=1, shuffle=False)
val_input_dataloader = DataLoader(val_input_dataset, batch_size=1, shuffle=False)
test_input_dataloader = DataLoader(test_input_dataset, batch_size=1, shuffle=False)
# Define the training, validation and test target dataloaders
train_target_dataloader = DataLoader(train_target_dataset, batch_size=1, shuffle=False)
val_target_dataloader = DataLoader(val_target_dataset, batch_size=1, shuffle=False)
test_target_dataloader = DataLoader(test_target_dataset, batch_size=1, shuffle=False)

# Define your LSTMModel, criterion, and optimizer as before
model = CCModel(input_size, 1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Check for an existing checkpoint file
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint'+datatype+'.pth')
if os.path.exists(checkpoint_path):
    # Load checkpoint weights and optimizer state
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")

def train_new():
    # Freeze all weights but that of the last layer, and set the model to training mode
    # Train only the last layer for a few epochs
    for param in model.parameters():
        param.requires_grad = False
        model.output_layer.weight.requires_grad = True
        model.output_layer.bias.requires_grad = True
    model.train()
    # Training loop
    last_validation_loss = None
    log_frequency = 10
    patience = 5  # Number of epochs to wait before stopping training if validation loss doesn't improve
    patience_counter = 0  # Counter to keep track of the number of epochs without improvement
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Reset the optimizer gradients
        for input_sequence, target_sequence in zip(train_input_dataloader, train_target_dataloader):
            # Forward pass with the input sequence
            outputs = model(input_sequence)
            # Compute the loss using the target sequence
            loss = criterion(outputs, target_sequence)
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            validation_loss = 0.0
            for input_sequence, target_sequence in zip(val_input_dataloader, val_target_dataloader):
                # Forward pass with the input sequence
                outputs = model(input_sequence)
                # Compute the loss using the target sequence
                loss = criterion(outputs, target_sequence)
                validation_loss += loss.item()
        if last_validation_loss is None or validation_loss < last_validation_loss:
            last_validation_loss = validation_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping due to lack of improvement in validation loss.")
            break
        if (epoch + 1) % checkpoint_frequency == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
        if (epoch + 1) % log_frequency == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {validation_loss/len(val_input_dataloader):.4f}')
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.4f} seconds")
    return loss.item()

loss_flag = True
start_time = time.time()
train_new()
for epoch in range(num_epochs_tot):
    model.add_layer()
    l = train_new()
    # Check loss
    if l < loss_threshold:
        loss_flag = False
        end_time = time.time()
        print(f"Training time for complete model: {end_time - start_time:.4f} seconds")
        break

# Test loop
model.eval()  # Set the model to evaluation mode
test_loss_sum = 0.0  # Initialize the sum of test losses
predictions, actuals = [], []
with torch.no_grad():
    for input_sequence, target_sequence in zip(test_input_dataloader, test_target_dataloader):
        # Forward pass with the input sequence
        test_outputs = model(input_sequence)
        # Compute the loss using the target sequence
        test_loss = criterion(test_outputs, target_sequence)
        test_loss_sum += test_loss.item()
        # Append predictions and actuals for metrics calculation
        predictions.append(test_outputs.detach().numpy())
        actuals.append(target_sequence.detach().numpy())
# Calculate the average test loss
average_test_loss = test_loss_sum / len(test_input_dataloader)
# Convert lists to numpy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)
# Calculate RMSE, MAE, MSE, R-squared
rmse = np.sqrt(((predictions - actuals) ** 2).mean())
mae = np.abs(predictions - actuals).mean()
mse = ((predictions - actuals) ** 2).mean()
r2 = r2_score(actuals, predictions)
print(f'Average Test Loss: {average_test_loss:.4f}')
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2 Score): {r2:.4f}')
print(f'Model size: {len(model.lstm_layers)}')