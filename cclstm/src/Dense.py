import torch
import torch.nn as nn

class DenseLSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseLSTMNetwork, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        lstm_layers = []
        for i in range(1, len(hidden_sizes)):
            lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
        self.lstm_layers = nn.ModuleList(lstm_layers)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out = torch.relu(self.input_layer(x))

        for lstm_layer in self.lstm_layers:
            out, _ = lstm_layer(out)

        out = self.output_layer(out[:, -1, :])  # Use only the last timestep's output

        return out

# Define the input size, hidden layer sizes, and output size
input_size = 784  # Example for a 28x28 pixel image
hidden_sizes = [128, 64, 32]  # Three hidden layers with different LSTM units
output_size = 10  # Number of classes for classification

# Create an instance of the dense LSTM neural network
model = DenseLSTMNetwork(input_size, hidden_sizes, output_size)

# Print the model architecture
print(model)
