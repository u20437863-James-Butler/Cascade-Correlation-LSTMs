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
    def __init__(self, input_size, output_size):
        super(CCModel, self).__init__()
        # LSTM layers
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, 1, 1, batch_first=True)])
        self.tot_size = input_size + 1
        # Output layer
        self.output_layer = nn.Linear(self.tot_size, output_size)
        self.input_size = input_size

    def forward(self, x): # x is pytorch tensor
        lstm_outputs = x.clone()
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(lstm_outputs)
            # x = nn.ReLU()(x)
            lstm_outputs = torch.cat((lstm_outputs, x), dim=2)
        # Use only the last timestep's output
        x = self.output_layer(lstm_outputs[:, -1, :])
        return x
    
    def add_layer(self):
        # for lstm_layer in self.lstm_layers:
        #     for param in lstm_layer.parameters():
        #         param.requires_grad = False
        self.tot_size = self.lstm_layers[-1].input_size + 1
        self.lstm_layers.append(nn.LSTM(self.tot_size, 1, 1, batch_first=True))
        self.lstm_layers[-1].flatten_parameters()
        # Rebuild linear layer with the updated tot_size
        linear_layer = nn.Linear(self.tot_size + 1, 1)
        # Initialize weights and bias for the new features
        new_weights = self.output_layer.weight.data.new_zeros((1, 1))
        # Concatenate the existing weights and bias with the new ones
        linear_layer.weight.data = torch.cat((self.output_layer.weight.data, new_weights), dim=1)
        linear_layer.bias.data = self.output_layer.bias.data.clone()
        self.output_layer = linear_layer
        print(f'Added layer. New neuron count: {self.tot_size - self.input_size + 1}')


# # Define a directory to save checkpoints
# checkpoint_dir = 'checkpoints_new'
# os.makedirs(checkpoint_dir, exist_ok=True)
# # Define the frequency of saving checkpoints (e.g., every 10 epochs)
# checkpoint_frequency = 20

# Choose which dataset to use (covid, stock, or wind)
datatype = input("Choose a dataset (covid, nyse, or wind): ")
if datatype == "covid": #use these values for multirun maybe
    input_size = 36
    learning_rate = 0.00001
    num_epochs = 10
    target = 'new_deaths_per_million'
    num_epochs_tot = 50
    batch_size = 32
    scaler = read_pkl_with_relative_path('../data/new_deaths_per_million_covid_scaler.pkl')
    patience = 5  # Number of epochs to wait before stopping training if validation loss doesn't improve
    big_patience_limit = 5
elif datatype == "nyse":
    input_size = 6
    # hidden_size = 10
    # num_layers = 10
    learning_rate = 0.00001
    num_epochs = 10
    target = 'close'
    # loss_threshold = 0.001
    num_epochs_tot = 50
    batch_size = 32
    # Load scaler from json file
    scaler = read_pkl_with_relative_path('../data/close_nyse_scaler.pkl')
    patience = 5  # Number of epochs to wait before stopping training if validation loss doesn't improve
    big_patience_limit = 5
elif datatype == "wind":
    input_size = 2
    # hidden_size = 10
    # num_layers = 5
    learning_rate = 0.001
    num_epochs = 15
    target = 'Turbine174_Speed'
    # loss_threshold = 0.001
    num_epochs_tot = 50
    batch_size = 32
    # Load scaler from json file
    scaler = read_pkl_with_relative_path('../data/Turbine174_Speed_wind_scaler.pkl')
    patience = 5  # Number of epochs to wait before stopping training if validation loss doesn't improve
    big_patience_limit = 5

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

print("Build model...")
# Define your LSTMModel, criterion, and optimizer as before
model = CCModel(input_size, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Check for an existing checkpoint file
# Get last checkpoint file
# checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{datatype}.pth')
# if os.path.exists(checkpoint_path):
#     # Load checkpoint weights and optimizer state
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")

# Training loop
model.to(device)
print("Starting training loop...")
print('Current time: ', time.strftime("%H:%M:%S", time.localtime()))
start_time = time.time()
log_frequency = 1
last_validation_loss = float('inf')
last_validation_loss_big = float('inf')
big_patience = 0
chpt_dict = {}
chpt_count = 0
# chpt_dict_large = {}
# chpt_count_large = 0
previous_loss = float('inf')
previos_vloss = float('inf')
for i in range(num_epochs_tot):
    print('Current epoch: ', i)
    patience_counter = 0  # Counter to keep track of the number of epochs without improvement
    # Print for each layer if its weights are frozen or not
    # for j in range(len(model.lstm_layers)):
    #     print(f'Layer {j} requires grad: {model.lstm_layers[j].weight_ih_l0.requires_grad}')
    # print(f'Output layer requires grad: {model.output_layer.weight.requires_grad}')
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
        if (epoch + 1) % log_frequency == 0:
            print(f'Epoch [{i}/{num_epochs_tot}] Sub-epoch[{epoch}/{num_epochs}], Training Loss: {loss.item():.8f}, Validation Loss: {validation_loss:.8f}')
        # Early stopping
        if validation_loss < last_validation_loss:
            last_validation_loss = validation_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1
        # Store last 5 checkpoints, in case of overfitting
        chpt_dict[chpt_count % patience] = {'chkpt': model.state_dict(), 'vloss': validation_loss.item()}
        chpt_count += 1
        if patience_counter >= patience:
            print("Early stopping due to lack of improvement in validation loss on subepoch.")
            # Restore from best checkpoint in dict based on vloss
            best_chpt = chpt_dict[min(chpt_dict, key=lambda x: chpt_dict[x]['vloss'])]['chkpt']
            model.load_state_dict(best_chpt)
            break
    # Validation loop for main epoch
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
    # Early stopping
    if validation_loss < last_validation_loss_big:
        last_validation_loss_big = validation_loss
        big_patience = 0  # Reset counter if validation loss improves
    else:
        big_patience += 1
    # Store last 5 checkpoints, for roleback in case of overfitting
    # chpt_dict[chpt_count_large % big_patience_limit] = {'chkpt': model.state_dict(), 'vloss': validation_loss.item()}
    # chpt_count += 1
    if big_patience >= big_patience_limit:
        print("Early stopping due to lack of improvement in validation loss.")
        # Restore from best checkpoint in dict based on vloss
        # best_chpt = chpt_dict[min(chpt_dict, key=lambda x: chpt_dict[x]['vloss'])]['chkpt']
        # model.load_state_dict(best_chpt)
        break
    if loss.item() == previous_loss and validation_loss == previos_vloss:
        print("Early stopping due to lack of change in both training and validation loss.")
        break
    previous_loss = loss.item()
    previos_vloss = validation_loss
    # if (epoch + 1) % checkpoint_frequency == 0:
    #     # Save checkpoint
    #     checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{datatype}.pth')
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #     }, checkpoint_path)
    if epoch != num_epochs_tot - 1:
        model.add_layer()
        model.to(device)

end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

# Test loop
start_time = time.time()
model.eval()  # Set the model to evaluation mode
# test_loss_sum = 0.0  # Initialize the sum of test losses
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
print(f'Test time: {time.time() - start_time:.4f} seconds')
print(f'neuron count: {model.tot_size - input_size + 1}')
