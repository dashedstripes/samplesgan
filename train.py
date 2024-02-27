import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SlidingWindowDataset(Dataset):
    def __init__(self, directory, sequence_length=100, step_length=1):
        self.directory = directory
        self.sequence_length = sequence_length
        self.step_length = step_length
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".wav")
        ]
        self.samples_per_file = []
        self.total_samples = 0
        for file in self.files:
            rate, data = wavfile.read(file)
            num_samples = (len(data) - sequence_length) // step_length + 1
            self.samples_per_file.append(num_samples)
            self.total_samples += num_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.samples_per_file[file_idx]:
            idx -= self.samples_per_file[file_idx]
            file_idx += 1

        rate, data = wavfile.read(self.files[file_idx])
        start_pos = idx * self.step_length
        end_pos = start_pos + self.sequence_length

        sequence = data[start_pos:end_pos] / 32768.0
        target_sample = (data[end_pos] if end_pos < len(data) else 0)
        target_sample = target_sample / 32768.0

        sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0)
        target_tensor = torch.tensor(target_sample, dtype=torch.float)

        return sequence_tensor, target_tensor


class WaveNetModel(nn.Module):
    def __init__(self, num_channels, num_blocks, num_layers, kernel_size=2):
        super(WaveNetModel, self).__init__()
        # Initialize layers
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        self.dilated_convs = nn.ModuleList([])
        self.residual_convs = nn.ModuleList([])
        self.skip_convs = nn.ModuleList([])
        self.start_conv = nn.Conv1d(1, num_channels, 1)

        # Building the WaveNet layers
        for b in range(num_blocks):
            for n in range(num_layers):
                dilation = 2**n
                self.dilated_convs.append(
                    nn.Conv1d(
                        num_channels, num_channels, kernel_size, dilation=dilation
                    )
                )
                self.residual_convs.append(nn.Conv1d(num_channels, num_channels, 1))
                self.skip_convs.append(nn.Conv1d(num_channels, num_channels, 1))

        self.end_conv1 = nn.Conv1d(num_channels, num_channels, 1)
        self.end_conv2 = nn.Conv1d(num_channels, 1, 1)

    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = []

        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                residual = x
                filtered = torch.tanh(self.dilated_convs[b * self.num_layers + n](x))
                gated = torch.sigmoid(self.dilated_convs[b * self.num_layers + n](x))
                x = filtered * gated  # Correct gate mechanism
                x = self.residual_convs[b * self.num_layers + n](x)

                # Ensure that residual and x are of the same shape
                if x.size(2) < residual.size(2):
                    residual = residual[:, :, :x.size(2)]
                elif x.size(2) > residual.size(2):
                    # Option to pad if x is longer, though unlikely in this context
                    padding = x.size(2) - residual.size(2)
                    residual = F.pad(residual, (0, padding))

                x = x + residual  # Element-wise addition

                skip = self.skip_convs[b * self.num_layers + n](x)
                skip_connections.append(skip)

        x = torch.sum(torch.stack(skip_connections), 0)
        x = torch.relu(self.end_conv1(x))
        x = self.end_conv2(x)

        x = torch.mean(x, dim=2, keepdim=True)

        return x


directory = "training_data/processed"
dataset = SlidingWindowDataset(directory, sequence_length=100, step_length=1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = WaveNetModel(num_channels=1, num_blocks=1, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1
for epoch in range(num_epochs):
    for i, (sequences, targets) in enumerate(dataloader):
        sequences, targets = sequences.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)

        outputs = outputs.squeeze(1).squeeze(1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")

print("Finished Training")
# save the weights
torch.save(model.state_dict(), "wavenet_model.pth")