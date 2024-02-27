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
    def __init__(self):
        super(WaveNetModel, self).__init__()
    def forward(self, x):
        
        return x


directory = "training_data/test_processed"
dataset = SlidingWindowDataset(directory)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

model = WaveNetModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (sequences, targets) in enumerate(dataloader):
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}"
            )

print("Finished Training")
# save the weights
torch.save(model.state_dict(), "wavenet_model.pth")
