import torch
import torch.nn as nn

class DenseNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseNeuralNetwork, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i-1], hidden_sizes[i]) for i in range(1, len(hidden_sizes))
        ])

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out = torch.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            out = torch.relu(hidden_layer(out))

        out = self.output_layer(out)

        return out

# Define the input size, hidden layer sizes, and output size
input_size = 784  # Example for a 28x28 pixel image
hidden_sizes = [128, 64, 32]  # Three hidden layers with different neuron counts
output_size = 10  # Number of classes for classification

# Create an instance of the dense neural network
model = DenseNeuralNetwork(input_size, hidden_sizes, output_size)

# Print the model architecture
print(model)
