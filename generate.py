import torch
import numpy as np
import torch.nn as nn
from scipy.io.wavfile import write


class Generator(nn.Module):
    def __init__(self, input_dim=100, output_channels=1, output_length=16000):
        super(Generator, self).__init__()
        self.init_size = output_length // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, output_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        audio = self.conv_blocks(out)
        return audio

def generate_audio(model, device='cpu'):
    model.to(device)
    model.eval()
    
    # Generate noise vector z
    input_dim = 100  # Adjust this based on your model's input dimension
    z = torch.randn(1, input_dim, device=device)
    
    # Generate audio samples
    with torch.no_grad():
        generated_samples = model(z)
        
    # Move generated samples to CPU and convert to 1D numpy array
    generated_samples = generated_samples.squeeze().to('cpu').numpy()
    
    return generated_samples

def floats_to_wav(audio_data, sample_rate, file_path):
    # Scale the floats to the range of 16-bit integers
    int_data = np.array(audio_data, dtype=np.float32)
    int_data = np.int16(int_data * 32767)

    # Write the data to a WAV file
    write(file_path, sample_rate, int_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Generator()
model.to(device)

model_weights_path = "generator.pth"
state_dict = torch.load(model_weights_path, map_location=device)
model.load_state_dict(state_dict)

# generate 10 samples
for i in range(10):
    generated_audio = generate_audio(model, device=device)
    sample_rate = 16000  # Replace with your actual sample rate
    file_path = f"generated/generated_audio_{i}.wav"  # Replace with your desired file path
    audio_data = generated_audio.tolist()  # Replace 'generated_audio.tolist()' with your list of floats

    floats_to_wav(audio_data, sample_rate, file_path)