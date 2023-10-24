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
import csv

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_csv_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)
    file_path = os.path.join(script_directory, file_name)
    return pd.read_csv(file_path)

def read_pkl_with_relative_path(file_name):
    script_directory = os.path.dirname(__file__)
    file_path = os.path.join(script_directory, file_name)
    return joblib.load(file_path)

class CCModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CCModel, self).__init__()
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, 1, 1, batch_first=True)])
        self.tot_size = input_size + 1
        self.output_layer = nn.Linear(self.tot_size, output_size)
        self.input_size = input_size

    def forward(self, x):
        lstm_outputs = x.clone()
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(lstm_outputs)
            lstm_outputs = torch.cat((lstm_outputs, x), dim=2)
        x = self.output_layer(lstm_outputs[:, -1, :])
        return x
    
    def add_layer(self):
        self.tot_size = self.lstm_layers[-1].input_size + 1
        self.lstm_layers.append(nn.LSTM(self.tot_size, 1, 1, batch_first=True))
        self.lstm_layers[-1].flatten_parameters()
        linear_layer = nn.Linear(self.tot_size + 1, 1)
        new_weights = self.output_layer.weight.data.new_zeros((1, 1))
        linear_layer.weight.data = torch.cat((self.output_layer.weight.data, new_weights), dim=1)
        linear_layer.bias.data = self.output_layer.bias.data.clone()
        self.output_layer = linear_layer
        print(f'Added layer. New neuron count: {self.tot_size - self.input_size + 1}')

datatype = input("Choose a dataset (covid, nyse, or wind): ")
num_runs = int(input("How many runs? "))
if datatype == "covid":
    input_size = 36
    learning_rate = 0.00001
    num_epochs = 10
    target = 'new_deaths_per_million'
    num_epochs_tot = 50
    batch_size = 32
    scaler = read_pkl_with_relative_path('../data/new_deaths_per_million_covid_scaler.pkl')
    patience = 5
    big_patience_limit = 5
elif datatype == "nyse":
    input_size = 6
    learning_rate = 0.00001
    num_epochs = 10
    target = 'close'
    num_epochs_tot = 50
    batch_size = 32
    scaler = read_pkl_with_relative_path('../data/close_nyse_scaler.pkl')
    patience = 5
    big_patience_limit = 5
elif datatype == "wind":
    input_size = 2
    learning_rate = 0.001
    num_epochs = 15
    target = 'Turbine174_Speed'
    num_epochs_tot = 50
    batch_size = 32
    scaler = read_pkl_with_relative_path('../data/Turbine174_Speed_wind_scaler.pkl')
    patience = 5
    big_patience_limit = 5

dataset = read_csv_with_relative_path('../data/'+datatype+'_data_fixed.csv')
x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=[target]), dataset[target], test_size=0.2, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
x_train_tensor = x_train_tensor.view(-1,1,input_size)
y_train_tensor = y_train_tensor.view(-1,1)
x_val_tensor = x_val_tensor.view(-1,1,input_size)
y_val_tensor = y_val_tensor.view(-1,1)
x_test_tensor = x_test_tensor.view(-1,1,input_size)
y_test_tensor = y_test_tensor.view(-1,1)
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

directory = 'results'
base_csv = os.path.join(directory, 'new_results.csv')
os.makedirs(directory, exist_ok=True)
is_new_file = not os.path.exists(base_csv) or os.path.getsize(base_csv) == 0
with open(base_csv, 'a', newline='') as file:
    fieldnames = ['Data_Set', 'Timestamp', 'Train_time', 'RMSE', 'MAE', 'MSE', 'R2_Score', 'Test_time', 'Neuron_Count']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if is_new_file:
        writer.writeheader()
    
    for i in range(num_runs):
        run_dict = {}
        loss_dict = []
        model = CCModel(input_size, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        previous_loss = float('inf')
        previos_vloss = float('inf')
        for i in range(num_epochs_tot):
            print('Current epoch: ', i)
            patience_counter = 0
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                model.train()
                for input_sequence, target_sequence in train_loader:
                    input_sequence = input_sequence.to(device)
                    target_sequence = target_sequence.to(device)
                    outputs = model(input_sequence)
                    loss = criterion(outputs, target_sequence.view(-1,1))
                    loss.backward()
                    optimizer.step()
                    input_sequence = input_sequence.to('cpu')
                    target_sequence = target_sequence.to('cpu')
                model.eval()
                with torch.no_grad():
                    x_val_tensor = x_val_tensor.to(device)
                    y_val_tensor = y_val_tensor.to(device)
                    outputs = model(x_val_tensor)
                    validation_loss = criterion(outputs, y_val_tensor.view(-1,1))
                    x_val_tensor = x_val_tensor.to('cpu')
                    y_val_tensor = y_val_tensor.to('cpu')
                if (epoch + 1) % log_frequency == 0:
                    print(f'Epoch [{i}/{num_epochs_tot}] Sub-epoch[{epoch}/{num_epochs}], Training Loss: {loss.item():.8f}, Validation Loss: {validation_loss:.8f}')
                    loss_dict.append({'epoch': i, 'subepoch': epoch,'train_loss': loss.item(), 'val_loss': validation_loss.item()})
                if validation_loss < last_validation_loss:
                    last_validation_loss = validation_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                chpt_dict[chpt_count % patience] = {'chkpt': model.state_dict(), 'vloss': validation_loss.item()}
                chpt_count += 1
                if patience_counter >= patience:
                    print("Early stopping due to lack of improvement in validation loss on subepoch.")
                    best_chpt = chpt_dict[min(chpt_dict, key=lambda x: chpt_dict[x]['vloss'])]['chkpt']
                    model.load_state_dict(best_chpt)
                    break
            model.eval()
            with torch.no_grad():
                x_val_tensor = x_val_tensor.to(device)
                y_val_tensor = y_val_tensor.to(device)
                outputs = model(x_val_tensor)
                validation_loss = criterion(outputs, y_val_tensor.view(-1,1))
                x_val_tensor = x_val_tensor.to('cpu')
                y_val_tensor = y_val_tensor.to('cpu')
            if validation_loss < last_validation_loss_big:
                last_validation_loss_big = validation_loss
                big_patience = 0
            else:
                big_patience += 1
            if big_patience >= big_patience_limit:
                print("Early stopping due to lack of improvement in validation loss.")
                break
            if loss.item() == previous_loss and validation_loss == previos_vloss:
                print("Early stopping due to lack of change in both training and validation loss.")
                break
            previous_loss = loss.item()
            previos_vloss = validation_loss
            if epoch != num_epochs_tot - 1:
                model.add_layer()
                model.to(device)
            model.eval()
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.4f} seconds")
        run_dict['Data_Set'] = datatype
        run_dict['Timestamp'] = time.time()
        run_dict['Train_time'] = end_time - start_time
        csv_name = os.path.join(directory, datatype+'_new_loss_'+str(time.time())+'.csv')
        with open(csv_name, 'w', newline='') as file:
            fnames = ['epoch', 'subepoch', 'train_loss', 'val_loss']
            write = csv.DictWriter(file, fieldnames=fnames)
            write.writeheader()
            for row in loss_dict:
                write.writerow(row)
        start_time = time.time()
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            x_test_tensor = x_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            test_outputs = model(x_test_tensor)
            x_test_tensor = x_test_tensor.to('cpu')
            y_test_tensor = y_test_tensor.to('cpu')
            y_test_np = y_test_tensor.numpy()
            test_predictions_np = test_outputs.cpu().numpy()
            test_predictions_np = scaler.inverse_transform(test_predictions_np)
            y_test_np = scaler.inverse_transform(y_test_np)
            mae = mean_absolute_error(y_test_np, test_predictions_np)
            mse = mean_squared_error(y_test_np, test_predictions_np)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_np, test_predictions_np)
        print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'R-squared (R2 Score): {r2:.4f}')
        end_time = time.time()
        print(f'Test time: {end_time - start_time:.4f} seconds')
        print(f'neuron count: {model.tot_size - input_size + 1}')
        run_dict['RMSE'] = rmse
        run_dict['MAE'] = mae
        run_dict['MSE'] = mse
        run_dict['R2_Score'] = r2
        run_dict['Test_time'] = end_time - start_time
        run_dict['Neuron_Count'] = model.tot_size - input_size + 1
        writer.writerow(run_dict)
