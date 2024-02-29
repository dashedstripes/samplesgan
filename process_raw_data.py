import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

def process_raw_data():
    raw_dir = "training_data/raw"
    processed_dir = "training_data/processed"
    os.makedirs(processed_dir, exist_ok=True)  # Ensure processed directory exists
    files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f)) and f.endswith(".wav")
    ]

    for i, f in enumerate(files):
        try:
            rate, data = wavfile.read(f)

            # Convert to mono if necessary
            if len(data.shape) > 1 and data.shape[1] > 1:
                # Averages the channels if there are more than one
                data = np.mean(data, axis=1)

            # Truncate or pad the data to make it 1 second long
            data = data[:rate]  # truncate to 1 second
            if len(data) < rate:
                data = np.pad(data, (0, rate - len(data)), "constant")

            # Resample to 16kHz if the original rate is different
            if rate != 16000:
                num_samples = int(16000 * (len(data) / rate))
                data = signal.resample(data, num_samples)

            # Normalize and convert data to 16-bit integer values
            data = np.int16(data / np.max(np.abs(data)) * 32767)
            data = np.clip(data, -32768, 32767)

            # make values between -1 and 1
            data = data / 32768.0

            wavfile.write(f"{processed_dir}/kick_{i}.wav", 16000, data)

        except Exception as e:
            print(f"Error processing {f}: {e}")

process_raw_data()

