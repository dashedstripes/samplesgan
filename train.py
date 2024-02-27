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
    def __init__(self, directory):
        self.directory = directory

    def __len__(self):
        for root, dirs, files in os.walk(self.directory):
            return len(files)

    def __getitem__(self, idx):
        print(idx)
        # rate, data = wavfile.read(f"{self.directory}/kick_{idx}.wav")

        return None


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
dataset = SlidingWindowDataset(directory)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

batch = next(iter(dataloader))

# # Initialize the model, loss function, and optimizer
# model = WaveNetModel(num_channels=1, num_blocks=4, num_layers=4)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# num_epochs = 1
# for epoch in range(num_epochs):
#     for i, (sequences, targets) in enumerate(dataloader):
#         sequences, targets = sequences.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(sequences)

#         outputs = outputs.squeeze(1).squeeze(1)

#         print(outputs)

        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")

# print("Finished Training")
# # save the weights
# torch.save(model.state_dict(), "wavenet_model.pth")