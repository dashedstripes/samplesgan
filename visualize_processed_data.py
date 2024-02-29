from scipy.io import wavfile
from matplotlib import pyplot as plt


def visualize_data():
    rate, data = wavfile.read("training_data/processed/kick_67.wav")
    plt.plot(data)
    plt.show()

visualize_data()
