import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import matplotlib.pyplot as plt


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
        target_sample = (
            data[end_pos] if end_pos < len(data) else 0
        ) / 32768.0

        sequence_tensor = torch.from_numpy(sequence).float()
        target_tensor = torch.tensor(target_sample, dtype=torch.float)

        return sequence_tensor, target_tensor


directory = "training_data/processed"
dataset = SlidingWindowDataset(directory, sequence_length=100, step_length=1)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False) # setting to false to make it easier to visualize the data

def visualize_sample():
    # Squeeze the batch dimension since batch_size=1 for plotting
    input_sequence_np = input_sequence.squeeze().numpy()  # Adjusted line

    plt.figure(figsize=(10, 4))
    plt.plot(input_sequence_np, label="Input Sequence")
    # Correctly adjust the index for the target sample and convert it to a numpy scalar
    plt.scatter(
        len(input_sequence_np) - 1,
        target_sample.numpy().squeeze(),
        color="red",
        label="Target Sample",
    )  # Adjusted line
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Normalized Audio Sequence with Target Sample")
    plt.legend()
    plt.grid(True)
    plt.show()

for input_sequence, target_sample in data_loader:
    print(input_sequence.shape, target_sample.shape)
    print(input_sequence, target_sample)
    visualize_sample()
    break
