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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def network(datatype, num_runs):
    if datatype == "covid":
        input_size = 36
        hidden_size = 8
        num_layers = 4
        learning_rate = 0.001
        num_epochs = 50
        target = 'new_deaths_per_million'
        batch_size = 64
        scaler = read_pkl_with_relative_path('../data/new_deaths_per_million_covid_scaler.pkl')
    elif datatype == "nyse":
        input_size = 6
        hidden_size = 8
        num_layers = 3
        learning_rate = 0.001
        num_epochs = 50
        target = 'close'
        batch_size = 64
        scaler = read_pkl_with_relative_path('../data/close_nyse_scaler.pkl')
    elif datatype == "wind":
        input_size = 2
        hidden_size = 10
        num_layers = 5
        learning_rate = 0.001
        num_epochs = 50
        target = 'Turbine174_Speed'
        batch_size = 32
        scaler = read_pkl_with_relative_path('../data/Turbine174_Speed_wind_scaler.pkl')

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

    directory = 'results'
    base_csv = os.path.join(directory, 'base_results.csv')
    os.makedirs(directory, exist_ok=True)
    is_new_file = not os.path.exists(base_csv) or os.path.getsize(base_csv) == 0
    with open(base_csv, 'a', newline='') as file:
        fieldnames = ['Data_Set', 'Timestamp', 'Train_time', 'RMSE', 'MAE', 'MSE', 'R2_Score', 'Test_time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()

        for i in range(num_runs):
            run_dict = {}
            loss_dict = []
            model = LSTMModel(input_size, hidden_size, num_layers, 1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.to(device)
            print("Starting training loop...")
            print('Current time: ', time.strftime("%H:%M:%S", time.localtime()))
            start_time = time.time()
            last_validation_loss = float('inf')
            log_frequency = 1
            patience = 10
            patience_counter = 0
            chpt_counter = 0
            chpt_dict = {}
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
                if validation_loss < last_validation_loss:
                    last_validation_loss = validation_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                chpt_dict[chpt_counter % patience] = {'chkpt': model.state_dict(), 'vloss': validation_loss.item()}
                chpt_counter += 1
                if patience_counter >= patience:
                    print("Early stopping due to lack of improvement in validation loss.")
                    best_chpt = chpt_dict[min(chpt_dict, key=lambda x: chpt_dict[x]['vloss'])]['chkpt']
                    model.load_state_dict(best_chpt)
                    break
                if (epoch + 1) % log_frequency == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.8f}, Validation Loss: {validation_loss:.8f}')
                    loss_dict.append({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': validation_loss.item()})
            end_time = time.time()
            print(f"Training time: {end_time - start_time:.4f} seconds")
            run_dict['Data_Set'] = datatype
            run_dict['Timestamp'] = time.time()
            run_dict['Train_time'] = end_time - start_time
            csv_name = os.path.join(directory, datatype+'_base_loss_'+str(time.time())+'.csv')
            with open(csv_name, 'w', newline='') as file:
                fnames = ['epoch', 'train_loss', 'val_loss']
                write = csv.DictWriter(file, fieldnames=fnames)
                write.writeheader()
                for row in loss_dict:
                    write.writerow(row)
            start_time = time.time()
            model.eval()
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
            end_time = time.time()
            print(f'Root Mean Square Error (RMSE): {rmse:.8f}')
            print(f'Mean Absolute Error (MAE): {mae:.8f}')
            print(f'Mean Squared Error (MSE): {mse:.8f}')
            print(f'R-squared (R2 Score): {r2:.8f}')
            print(f'Test time: {end_time - start_time:.8f} seconds')
            run_dict['RMSE'] = rmse
            run_dict['MAE'] = mae
            run_dict['MSE'] = mse
            run_dict['R2_Score'] = r2
            run_dict['Test_time'] = end_time - start_time
            writer.writerow(run_dict)

mode = input("Choose a mode (select or all): ")
if mode == "select":
    datatype = input("Choose a dataset (covid, nyse, or wind): ")
    num_runs = int(input("How many runs? "))
    network(datatype, num_runs)
elif mode == "all":
    num_runs = int(input("How many runs? "))
    print("Wind dataset")
    network("wind", num_runs)
    print("NYSE dataset")
    network("nyse", num_runs)
    print("Covid dataset")
    network("covid", num_runs)
