import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Define a directory to save checkpoints
checkpoint_dir = 'checkpoints_base'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the frequency of saving checkpoints (e.g., every 10 epochs)
checkpoint_frequency = 10

# Define the original perceptron
class OriginalPerceptron(nn.Module):
    def __init__(self):
        super(OriginalPerceptron, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=2)
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Define the new model with added LSTM layers
class NewModelWithAddedLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NewModelWithAddedLSTM, self).__init__()
        
        # Original perceptron
        self.original_perceptron = OriginalPerceptron()

        # Fully connected LSTM layers
        lstm_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            lstm_layers.append(nn.LSTM(prev_size, hidden_size, batch_first=True))
            prev_size = hidden_size
        self.lstm_layers = nn.ModuleList(lstm_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x_original = self.original_perceptron(x)
        out = x_original

        for lstm_layer in self.lstm_layers:
            out, _ = lstm_layer(out)

        out = self.output_layer(out[:, -1, :])  # Use only the last timestep's output

        return out

# Instantiate the original perceptron and load its weights
original_perceptron = OriginalPerceptron()
original_weights = original_perceptron.state_dict()

# Number of new LSTM layers to add
num_new_lstm_layers = 3

previous_layer = original_perceptron.fc  # Initialize with the original layer

# Dummy data for demonstration
dummy_input = torch.randn(32, 10)

# Training loop for adding new layers
for i in range(num_new_lstm_layers):
    # Define the hidden sizes for LSTM layers
    hidden_sizes = [32] * (i + 1)  # Example: Using the same hidden size for all added LSTM layers
    
    # Instantiate the new model with added LSTM layer
    new_model = NewModelWithAddedLSTM(input_size=10, hidden_sizes=hidden_sizes, output_size=2)  # Adjust output_size as needed
    
    # Set requires_grad to False for the original layers
    original_perceptron.fc.requires_grad = False
    
    # Print the new model architecture
    print(f"Iteration {i+1}:")
    print(new_model)
    print("------------------------")
    
    # Training loop for the new LSTM layer
    # Define a dummy target for demonstration
    dummy_target = torch.randint(0, 2, (32,))
    
    # Define loss function and optimizer
    criterion_new = nn.CrossEntropyLoss()
    optimizer_new = optim.SGD(new_model.parameters(), lr=0.01)
    
    # Training loop for the new layer
    for epoch in range(10):  # Example: 10 epochs
        optimizer_new.zero_grad()
        outputs_new = new_model(dummy_input)
        loss_new = criterion_new(outputs_new, dummy_target)
        loss_new.backward()
        optimizer_new.step()
        
        print(f"New LSTM Layer - Epoch [{epoch+1}/10], Loss: {loss_new.item():.4f}")
    
    # Update original weights with the new model's weights
    original_weights = new_model.state_dict()
