import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, directory):
        self.audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        _, data = wavfile.read(self.audio_files[idx])
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data

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


class Discriminator(nn.Module):
    def __init__(self, input_length=16000):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 64, 3, stride=1, padding=1),
            nn.Conv1d(64, 32, 3, stride=2, padding=1),
            nn.Conv1d(32, 16, 3, stride=4, padding=1),
            nn.Conv1d(16, 8, 3, stride=8, padding=1),
            nn.Flatten(),
            nn.Linear(2000, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_sample):
        x = self.model(audio_sample)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

# hyperparameters
lr = 0.0002
batch_size = 32
epochs = 200
sample_interval = 50
input_dim = 100  # Dimension of the noise vector

criterion = nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

dataset = AudioDataset('./training_data/processed')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


for epoch in range(epochs):
    for i, real_samples in enumerate(train_loader):
        real_samples = real_samples.to(device)

        # Prepare real samples and fake samples
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)
        z = torch.randn((batch_size, input_dim), device=device)
        # # ---------------------
        # #  Train Discriminator
        # # ---------------------
        optimizer_D.zero_grad()
        
        # # Loss on real samples
        real_predictions = discriminator(real_samples)
        d_loss_real = criterion(real_predictions, real_labels)
        
        # # Loss on fake samples
        fake_samples = generator(z).detach()
        fake_predictions = discriminator(fake_samples)
        d_loss_fake = criterion(fake_predictions, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # # -----------------
        # #  Train Generator
        # # -----------------
        optimizer_G.zero_grad()
        
        # Generate a batch of samples
        fake_samples = generator(z)
        # Discriminator's prediction on fake samples
        validity = discriminator(fake_samples)
        # Loss measures generator's ability to fool the discriminator
        g_loss = criterion(validity, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
    # Print some progress every now and then
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs} | D loss: {d_loss.item()} | G loss: {g_loss.item()}")

# Save the model
torch.save(generator.state_dict(), 'generator.pth')