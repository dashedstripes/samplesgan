import torch
import numpy as np
import torch.nn as nn
from scipy.io.wavfile import write


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
                x = torch.tanh(
                    self.dilated_convs[b * self.num_layers + n](x)
                ) * torch.sigmoid(self.dilated_convs[b * self.num_layers + n](x))
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
    current_input = torch.rand(1, 1, 100).to(device) * 2 - 1

    generated_audio = []

    with torch.no_grad():  # No need to track gradients
        for _ in range(num_samples):
            # Forward pass through the model
            output = model(current_input)

            # Get the last output sample
            new_sample = output[:, :, -1].cpu().numpy()

            # Append the generated sample to the output list
            generated_audio.append(new_sample)

            # Update the current input (slide window and insert the new sample)
            current_input = torch.cat((current_input[:, :, 1:], output[:, :, -1:]), dim=2)

    # Convert the list of samples to a single numpy array and reshape it
    generated_audio = np.concatenate(generated_audio).reshape(-1)

    return generated_audio

def floats_to_wav(audio_data, sample_rate, file_path):
    # Scale the floats to the range of 16-bit integers
    int_data = np.array(audio_data, dtype=np.float32)
    int_data = np.int16(int_data * 32767)

    # Write the data to a WAV file
    write(file_path, sample_rate, int_data)

model = WaveNetModel(num_channels=1, num_blocks=1, num_layers=1)
model_weights_path = "wavenet_model.pth"
state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

generated_audio = generate_audio(model, sample_rate=16000, duration=1, device='cpu')

print(generated_audio)

# sample_rate = 16000  # Replace with your actual sample rate
# file_path = "generated/generated_audio.wav"  # Replace with your desired file path
# audio_data = generated_audio.tolist()  # Replace 'generated_audio.tolist()' with your list of floats

# floats_to_wav(audio_data, sample_rate, file_path)