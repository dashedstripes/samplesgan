import torch
import numpy as np
import torch.nn as nn
from scipy.io.wavfile import write


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
    current_input = torch.rand(1, 1, 99).to(device) * 2 - 1

    generated_audio = []

    with torch.no_grad():  # No need to track gradients
        for _ in range(num_samples):
            # Forward pass through the model
            output = model(current_input)

            # Get the last output sample (output is a single tensor (x))
            new_sample = output.item()
            print(current_input, new_sample)

            # Append the generated sample to the output list
            generated_audio.append(new_sample)


            # Update the current input (slide window and insert the new sample)
            current_input = torch.roll(current_input, shifts=-1, dims=2)

    # Convert the list of samples to a single numpy array and reshape it
    generated_audio = np.concatenate(generated_audio).reshape(-1)

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

print(generated_audio)

# sample_rate = 16000  # Replace with your actual sample rate
# file_path = "generated/generated_audio.wav"  # Replace with your desired file path
# audio_data = generated_audio.tolist()  # Replace 'generated_audio.tolist()' with your list of floats

# floats_to_wav(audio_data, sample_rate, file_path)