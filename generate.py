import torch
import numpy as np
import torch.nn as nn
from scipy.io.wavfile import write


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

def generate_audio(model, sample_rate=16000, duration=1, device='cuda'):
    """
    Generate audio using a trained WaveNet model.

    Args:
    - model: The trained WaveNet model.
    - sample_rate: The sample rate of the audio to generate.
    - duration: The duration of the audio to generate in seconds.
    - device: The device to run the generation on ('cuda' or 'cpu').

    Returns:
    - generated_audio: A numpy array containing the generated audio samples.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # Calculate the number of samples to generate
    num_samples = sample_rate * duration

    # Initialize the seed with zeros (or you could use random noise)
    current_input = torch.rand(1, 1, 99).to(device)

    generated_audio = []

    with torch.no_grad():  # No need to track gradients
        for _ in range(num_samples):
            # Forward pass through the model
            output = model(current_input)
            next_input = output.squeeze()
            # reshape next_input to match the input dimensions
            next_input = next_input.view(1, 1, 1)

            # Use the last output as the next input
            # Note: Ensure output is unsqueezed and matches the input dimensions
            current_input = torch.cat((current_input[:, :, 1:], next_input), dim=2)

            # Store the generated sample
            generated_audio.append(next_input.item())

    # Convert the list of samples to a single numpy array and reshape it
    generated_audio = np.array(generated_audio).reshape(-1)

    return generated_audio

def floats_to_wav(audio_data, sample_rate, file_path):
    # Scale the floats to the range of 16-bit integers
    int_data = np.array(audio_data, dtype=np.float32)
    int_data = np.int16(int_data * 32767)

    # Write the data to a WAV file
    write(file_path, sample_rate, int_data)

model = WaveNetModel()
model_weights_path = "wavenet_model.pth"
state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

generated_audio = generate_audio(model, sample_rate=16000, duration=1, device='cpu')

sample_rate = 16000  # Replace with your actual sample rate
file_path = "generated/generated_audio.wav"  # Replace with your desired file path
audio_data = generated_audio.tolist()  # Replace 'generated_audio.tolist()' with your list of floats

floats_to_wav(audio_data, sample_rate, file_path)