# Basic LSTM model
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

def read_csv_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)  # Get the directory of your script
    file_path = os.path.join(script_directory, file_name)
    return pd.read_csv(file_path)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):# this function is used to initialize the model
        super(LSTMModel, self).__init__() # this function is used to initialize the parent class
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # this function is used to initialize the LSTM layer
        self.fc = nn.Linear(hidden_size, output_size) # this function is used to initialize the fully connected layer
    
    def forward(self, x): # this function is used to define the forward pass
        print('x',x.shape)
        out, _ = self.lstm(x) # Get the output and hidden state of the last layer
        print('out',out.shape)
        out = out[:, -1, :]
        print('out',out.shape)
        out = self.fc(out)  # Get the last time step's output
        return out # return the output

# Define a directory to save checkpoints
checkpoint_dir = 'checkpoints_new'
os.makedirs(checkpoint_dir, exist_ok=True)
# Define the frequency of saving checkpoints (e.g., every 10 epochs)
checkpoint_frequency = 20

# Choose which dataset to use (covid, stock, or wind)
datatype = input("Choose a dataset (covid, stock, or wind): ")
if datatype == "covid":
    input_size = 37
    hidden_size = 2
    num_layers = 1
    learning_rate = 0.01
    num_epochs = 10
    target = 'new_deaths_per_million'
    loss_threshold = 0.1
    num_epochs_tot = 1000
elif datatype == "stock":
    input_size = 6
    hidden_size = 2
    num_layers = 1
    learning_rate = 0.01
    num_epochs = 10
    target = 'close'
    loss_threshold = 0.1
    num_epochs_tot = 1000
elif datatype == "wind":
    input_size = 2
    hidden_size = 32
    num_layers = 1
    learning_rate = 0.01
    num_epochs = 10
    target = 'Turbine174_Speed'
    loss_threshold = 0.1
    num_epochs_tot = 1000


# Load your dataset (replace 'datatype' with the appropriate dataset type)
dataset = read_csv_with_relative_path('../data/'+datatype+'_data_fixed.csv')
# Split the dataset into train, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=[target]), dataset[target], test_size=0.2, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

# Convert the train, validation, and test sets to PyTorch dataloaders

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

# Define your LSTMModel, criterion, and optimizer as before
model = LSTMModel(input_size, hidden_size, num_layers, 1)
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

# Training loop
print("Starting training loop...")
start_time = time.time()
last_validation_loss = float('inf')
log_frequency = 10
patience = 5  # Number of epochs to wait before stopping training if validation loss doesn't improve
patience_counter = 0  # Counter to keep track of the number of epochs without improvement
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    # optimizer.zero_grad()  # Reset the optimizer gradients
    for input_sequence, target_sequence in train_loader:
        print('Input:',input_sequence)
        outputs = model(input_sequence)
        target_sequence = target_sequence.view(-1,1)
        loss = criterion(outputs, target_sequence)
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        # Forward pass with the input sequence
        outputs = model(x_val_tensor)
        # Compute the loss using the target sequence
        validation_loss = criterion(outputs, y_val_tensor.view(-1,1))
    if validation_loss < last_validation_loss:
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {validation_loss:.4f}')
end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

# Test loop
model.eval()  # Set the model to evaluation mode
test_loss_sum = 0.0  # Initialize the sum of test losses
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
predictions, actuals = [], []
with torch.no_grad():
    # Forward pass with the input sequence
    test_outputs = model(x_test_tensor)
    y_test_np = y_test_tensor.numpy()
    test_predictions_np = test_outputs.numpy()
    mae = mean_absolute_error(y_test_np, test_predictions_np)
    mse = mean_squared_error(y_test_np, test_predictions_np)
    rmse = np.sqrt(mse)
    r2 = np.corrcoef(y_test_np, test_predictions_np, rowvar=False)[0, 1] ** 2
# Calculate RMSE, MAE, MSE, R-squared
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2 Score): {r2:.4f}')

