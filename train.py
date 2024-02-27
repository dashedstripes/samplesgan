import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import torch.nn as nn
import torch.optim as optim


class SlidingWindowDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory

    def __len__(self):
        for _, _, files in os.walk(self.directory):
          return len(files) - 2

    def __getitem__(self, idx):
        _, data = wavfile.read(f"{self.directory}/chunk_{idx}.wav")

        sequence = data[0:99]
        target = data[99]

        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32)

        return sequence, target


class WaveNetModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_blocks=1, num_layers=10, kernel_size=1):
        super(WaveNetModel, self).__init__()

        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=2**layer)
            for _ in range(num_blocks) for layer in range(num_layers)
        ])

        self.residual_convs = nn.ModuleList([
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
            for _ in range(num_blocks * num_layers)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
            for _ in range(num_blocks * num_layers)
        ])

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        skip_connections = []

        for dilated_conv, residual_conv, skip_conv in zip(self.dilated_convs, self.residual_convs, self.skip_convs):
            out = dilated_conv(x)
            skip = skip_conv(out)
            skip_connections.append(skip)

            out = residual_conv(out)
            x = out + x

        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = self.final_conv(x)
        x = x.squeeze()

        return x


directory = "training_data/processed"
dataset = SlidingWindowDataset(directory)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

model = WaveNetModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1
for epoch in range(num_epochs):
    for i, (sequences, targets) in enumerate(dataloader):
        sequences, targets = sequences.to(device), targets.to(device)
        print(targets)
        # optimizer.zero_grad()
        # outputs = model(sequences)
        # print(outputs)

        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print(
        #         f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}"
        #     )

# print("Finished Training")
# # save the weights
# torch.save(model.state_dict(), "wavenet_model.pth")
