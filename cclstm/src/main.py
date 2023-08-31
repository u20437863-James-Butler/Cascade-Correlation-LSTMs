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

# Define the original neural network
class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the original model and load its weights
original_model = BasicNN()
original_weights = original_model.state_dict()

# Number of new layers to add
num_new_layers = 3

# Dummy data for demonstration
dummy_input = torch.randn(32, 10)

# Training the original model's weights before adding new layers
# Define a dummy target for demonstration
dummy_target = torch.randint(0, 2, (32,))  # Assuming binary classification
    
# Define loss function and optimizer
criterion_original = nn.CrossEntropyLoss()
optimizer_original = optim.SGD(original_model.parameters(), lr=0.01)

# Training loop for original model
for epoch in range(10):  # Example: 10 epochs
    optimizer_original.zero_grad()
    outputs_original = original_model(dummy_input)
    loss_original = criterion_original(outputs_original, dummy_target)
    loss_original.backward()
    optimizer_original.step()
    
    print(f"Original Model - Epoch [{epoch+1}/10], Loss: {loss_original.item():.4f}")

# Training loop for adding new layers
previous_layer = original_model.fc2  # Initialize with the original last layer

for i in range(num_new_layers):
    class CCNN(nn.Module):
        def __init__(self, original_weights, previous_layer):
            super(CCNN, self).__init__()
            self.previous_layer = previous_layer  # Store previous layer
            
            self.fc1 = nn.Linear(in_features=10, out_features=5)
            self.fc2 = nn.Linear(in_features=5, out_features=2)
            
            self.additional_layer = nn.Linear(in_features=2, out_features=3)  # Adding a new layer
            
            # Load weights from the previous model
            self.fc1.load_state_dict(original_weights['fc1'])
            self.fc2.load_state_dict(original_weights['fc2'])
            
            self.load_state_dict(original_weights)  # Load original weights
            
            self.previous_layer = self.additional_layer  # Update previous layer
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            x = torch.relu(self.previous_layer(x))  # Pass through all layers
            return x
    
    # Instantiate the new model with the added layer and original weights
    new_model = CCNN(original_weights, previous_layer)
    
    # Set requires_grad to False for the previous layers
    new_model.fc1.requires_grad = False
    new_model.fc2.requires_grad = False
    
    # Update the previous layer for the next iteration
    previous_layer = new_model.previous_layer
    
    # Print the new model architecture
    print(f"Iteration {i+1}:")
    print(new_model)
    print("------------------------")
    
    # Training loop for the newly added layer
    # Define a dummy target for demonstration
    dummy_target = torch.randint(0, 3, (32,))
    
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
        
        print(f"New Layer - Epoch [{epoch+1}/10], Loss: {loss_new.item():.4f}")
    
    # Update original weights with the new model's weights
    original_weights = new_model.state_dict()
