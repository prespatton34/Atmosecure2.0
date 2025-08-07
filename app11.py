import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define wealth wave function
def wealth_wave(t, freq, phase_shift=0):
    return torch.sin(2 * np.pi * freq * t + phase_shift)

# Neural Network class representing brain signals directed to nerves with VPN protection
class WealthBrainModel(nn.Module):
    def __init__(self):
        super(WealthBrainModel, self).__init__()
        # Define layers of the network
        self.fc1 = nn.Linear(1, 64)  # Input layer (brain)
        self.fc2 = nn.Linear(64, 64)  # Hidden layer (signal propagation)
        self.fc3 = nn.Linear(64, 64)  # Storage layer (wealth data stored in nerves)
        self.fc_vpn = nn.Linear(64, 64)  # VPN protection layer
        self.fc4 = nn.Linear(64, 1)   # Pulse layer (output pulse representing stored data)

    def forward(self, x):
        # Wealth signal propagation through layers
        x = torch.relu(self.fc1(x))  # Brain layer
        x = torch.relu(self.fc2(x))  # Signal propagation layer
        stored_data = torch.relu(self.fc3(x))  # Store data in the nerves
        
        # VPN protection layer: Protect the stored wealth data
        protected_data = torch.relu(self.fc_vpn(stored_data))  # Data is encrypted and protected here
        
        # Generate pulse signal based on protected data
        pulse_signal = torch.sigmoid(self.fc4(protected_data))  
        return pulse_signal, protected_data

# Initialize the model
model = WealthBrainModel()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Time steps and frequencies for the wealth waves
time_steps = torch.linspace(0, 10, 1000)
freq_alpha = 10  # Alpha frequency (10 Hz)
freq_beta = 20   # Beta frequency (20 Hz)
freq_gamma = 40  # Gamma frequency (40 Hz)

# Simulate a continuous loop of wealth wave propagation
stored_data_all = []
for epoch in range(100):  # Simulate over 100 epochs (continuous propagation)
    model.train()
    
    # Generate wealth waves with phase shifts
    wealth_alpha = wealth_wave(time_steps, freq_alpha, phase_shift=epoch)
    wealth_beta = wealth_wave(time_steps, freq_beta, phase_shift=epoch + 0.5)
    wealth_gamma = wealth_wave(time_steps, freq_gamma, phase_shift=epoch + 1)
    
    # Combine signals (multi-layered wealth wave)
    wealth_input = wealth_alpha + wealth_beta + wealth_gamma
    wealth_input = wealth_input.unsqueeze(1)  # Reshape for model input
    
    # Forward pass through the network (brain -> nerves -> VPN -> stored pulse)
    pulse_signal, protected_data = model(wealth_input)
    
    # Store the protected data for analysis
    stored_data_all.append(protected_data.detach().numpy())
    
    # Simulate intruders (random noise) trying to tamper with the data
    intruder_noise = torch.randn_like(pulse_signal) * 0.1  # Small noise signal
    corrupted_pulse = pulse_signal + intruder_noise  # Intruder tries to corrupt the pulse
    
    # Compute loss based on how well the VPN layer protects from noise
    loss = criterion(corrupted_pulse, pulse_signal)  # Aim to protect pulse from noise
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Plot the pulse signal at every few steps to visualize protection
    if epoch % 10 == 0:
        plt.plot(time_steps.numpy(), pulse_signal.detach().numpy(), label=f'Epoch {epoch}')
        
plt.title("Atmosecure 2.0")
plt.xlabel("Time")
plt.ylabel("Pulse Signal")
plt.legend()
plt.show()

# Visualize protected wealth data over time
plt.imshow(np.mean(np.array(stored_data_all), axis=0), aspect='auto', cmap='viridis') # Average across the first axis to get a 2D array
plt.colorbar(label="Protected Wealth Data in Nerves")
plt.xlabel("Epochs")
plt.ylabel("Nerve Data Points")
plt.title("Atmosecure 2.0")
plt.show()