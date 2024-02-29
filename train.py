import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SlidingWindowDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory

    def __len__(self):
        for _, _, files in os.walk(self.directory):
          return len(files)

    def __getitem__(self, idx):
        _, data = wavfile.read(f"{self.directory}/chunk_{idx}.wav")

        sequence = data[0:99]
        target = data[99]

        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32)

        return sequence, target

class AttentionModel(nn.Module):
    def __init__(self, sequence_length, attention_size):
        super(AttentionModel, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(64, 50, batch_first=True)
        self.attention_weights = nn.Parameter(torch.randn(attention_size, sequence_length))
        self.dense1 = nn.Linear(50 + attention_size, 50)
        self.dense2 = nn.Linear(50, 1)
    
    def forward(self, x):
        # Convolutional part
        # x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))

        x = x.permute(0, 2, 1)
        
        # # LSTM part
        lstm_out, (hn, cn) = self.lstm(x)

        
        lstm_out = lstm_out.transpose(1, 2)
        print(self.attention_weights.shape)
        print(lstm_out.shape)
        
        # # Attention mechanism
        attention_scores = F.softmax(torch.matmul(self.attention_weights, lstm_out.transpose(1, 2)), dim=2)
        
        weighted_features = torch.matmul(attention_scores, lstm_out)
        
        # # Combining LSTM output with attention weighted features
        combined_features = torch.cat((lstm_out[:, -1, :], weighted_features.squeeze(1)), dim=1)
        
        # # Passing through dense layers
        x = F.relu(self.dense1(combined_features))
        # x = self.dense2(x)
        
        return x

class WaveNetModel(nn.Module):
    def __init__(self):
        super(WaveNetModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.dense1 = nn.Linear(50, 50)
        self.dense2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]

        x = self.relu(self.dense1(x))
        x = self.dense2(x)

        return x


directory = "training_data/processed"
dataset = SlidingWindowDataset(directory)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

sequence_length = 100
attention_size = 4
model = AttentionModel(sequence_length, attention_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 100
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
