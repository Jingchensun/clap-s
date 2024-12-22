import torch
import torch.nn as nn

# Define Adapter model
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the Adapter model
model = Adapter(1024, 4)

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count the number of parameters in the Adapter model
adapter_params = count_parameters(model)

# Calculate storage size
param_size_in_bytes = adapter_params * 4  # Each parameter is 4 bytes (float32)
param_size_in_bits = param_size_in_bytes * 8
param_size_in_mb = param_size_in_bytes / (1024 * 1024)

print(f'Adapter Model Parameters: {adapter_params}')
print(f'Total Size: {param_size_in_bits} bits')
print(f'Total Size: {param_size_in_bytes} bytes')
print(f'Total Size: {param_size_in_mb:.2f} MB')
