import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import json
import joblib

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_csv_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)  # Get the directory of your script
    file_path = os.path.join(script_directory, file_name)
    return pd.read_csv(file_path)

def read_pkl_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)  # Get the directory of your script
    file_path = os.path.join(script_directory, file_name)
    return joblib.load(file_path)

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
        print(x)
        out = []
        out = out + x.tolist()
        # For each layer, add all outputs to the output list
        # which is passed as the input into the next layer
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(torch.tensor(out, dtype=torch.float32))
            out.append(x.tolist())
        # Use only the last timestep's output
        x = self.output_layer(torch.tensor(out, dtype=torch.float32)[:, -1, :])
        return x
    
    def add_layer(self):
        self.lstm_layers.append(nn.LSTMCell(self.prev_size, 1, batch_first=True))
        self.tot_size = self.tot_size + 1
        # Rebuild linear layer
        linear_layer = nn.Linear(self.tot_size, 1)
        linear_layer.weight.data[:-1] = self.output_layer.weight.data
        linear_layer.bias.data[:-1] = self.output_layer.bias.data
        self.output_layer = linear_layer

# Define a directory to save checkpoints
checkpoint_dir = 'checkpoints_new'
os.makedirs(checkpoint_dir, exist_ok=True)
# Define the frequency of saving checkpoints (e.g., every 10 epochs)
checkpoint_frequency = 20

# Choose which dataset to use (covid, stock, or wind)
datatype = input("Choose a dataset (covid, nyse, or wind): ")
if datatype == "covid":
    input_size = 36
    hidden_size = 10
    num_layers = 8
    learning_rate = 0.01
    num_epochs = 100
    target = 'new_deaths_per_million'
    loss_threshold = 0.001
    num_epochs_tot = 1000
    batch_size = 32
    # Load scaler from json file
    scaler = read_pkl_with_relative_path('../data/new_deaths_per_million_covid_scaler.pkl')
elif datatype == "nyse":
    input_size = 6
    hidden_size = 10
    num_layers = 10
    learning_rate = 0.01
    num_epochs = 100
    target = 'close'
    loss_threshold = 0.001
    num_epochs_tot = 1000
    batch_size = 32
    # Load scaler from json file
    scaler = read_pkl_with_relative_path('../data/close_nyse_scaler.pkl')
elif datatype == "wind":
    input_size = 2
    hidden_size = 10
    num_layers = 5
    learning_rate = 0.01
    num_epochs = 100
    target = 'Turbine174_Speed'
    loss_threshold = 0.001
    num_epochs_tot = 1000
    batch_size = 2
    # Load scaler from json file
    scaler = read_pkl_with_relative_path('../data/Turbine174_Speed_wind_scaler.pkl')

# Load your dataset (replace 'datatype' with the appropriate dataset type)
dataset = read_csv_with_relative_path('../data/'+datatype+'_data_fixed.csv')
# Split the dataset into train, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=[target]), dataset[target], test_size=0.2, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
# Convert the train, validation, and test sets to PyTorch dataloaders
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
# Reshape
x_train_tensor = x_train_tensor.view(-1,1,input_size) # reshape to (batch_size, sequence_length, input_size)
y_train_tensor = y_train_tensor.view(-1,1)
x_val_tensor = x_val_tensor.view(-1,1,input_size)
y_val_tensor = y_val_tensor.view(-1,1)
x_test_tensor = x_test_tensor.view(-1,1,input_size)
y_test_tensor = y_test_tensor.view(-1,1)
# Dataloaders
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Define your LSTMModel, criterion, and optimizer as before
model = CCModel(input_size, hidden_size, 1)
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

model.to(device)
# Training loop
print("Starting training loop...")
print('Current time: ', time.strftime("%H:%M:%S", time.localtime()))
start_time = time.time()
last_validation_loss = float('inf')
log_frequency = 1
patience = 5  # Number of epochs to wait before stopping training if validation loss doesn't improve
patience_counter = 0  # Counter to keep track of the number of epochs without improvement
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset the optimizer gradients
    model.train()  # Set the model to training mode
    for input_sequence, target_sequence in train_loader:
        input_sequence = input_sequence.to(device)
        target_sequence = target_sequence.to(device)
        outputs = model(input_sequence)  # Forward pass
        # y_train_tensor = y_train_tensor.view(-1,1)
        loss = criterion(outputs, target_sequence.view(-1,1))  # Compute the Loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        input_sequence = input_sequence.to('cpu')
        target_sequence = target_sequence.to('cpu')
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        x_val_tensor = x_val_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)
        # Forward pass with the input sequence
        outputs = model(x_val_tensor)
        # Compute the loss using the target sequence
        validation_loss = criterion(outputs, y_val_tensor.view(-1,1))
        x_val_tensor = x_val_tensor.to('cpu')
        y_val_tensor = y_val_tensor.to('cpu')
    if validation_loss < last_validation_loss:
        last_validation_loss = validation_loss
        patience_counter = 0  # Reset counter if validation loss improves
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping due to lack of improvement in validation loss.")
        break
    if loss.item() < loss_threshold:
        print("Loss threshold reached. Stopping training.")
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {validation_loss:.4f}')

end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

# Test loop
model.eval()  # Set the model to evaluation mode
test_loss_sum = 0.0  # Initialize the sum of test losses
predictions, actuals = [], []
with torch.no_grad():
    x_test_tensor = x_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    # Forward pass with the input sequence
    test_outputs = model(x_test_tensor)
    x_test_tensor = x_test_tensor.to('cpu')
    y_test_tensor = y_test_tensor.to('cpu')
    y_test_np = y_test_tensor.numpy()
    test_predictions_np = test_outputs.cpu().numpy()
    # Scale back to the original range
    test_predictions_np = scaler.inverse_transform(test_predictions_np)
    y_test_np = scaler.inverse_transform(y_test_np)
    mae = mean_absolute_error(y_test_np, test_predictions_np)
    mse = mean_squared_error(y_test_np, test_predictions_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, test_predictions_np)
# Calculate RMSE, MAE, MSE, R-squared
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2 Score): {r2:.4f}')
