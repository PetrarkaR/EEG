import mne
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=sys.maxsize)

def load(file):
    # Load EDF file and get raw data and sampling frequency
    data = mne.io.read_raw_edf(file, preload=True)
    raw_data = data.get_data()
    channels = data.ch_names
    sfreq = data.info['sfreq']  # Get sampling frequency
    channel_data_dict = {channels[i]: raw_data[i] for i in range(len(channels))}
    print(f"Channels loaded: {channels}, Sampling frequency: {sfreq} Hz")
    return channel_data_dict, sfreq

def plot_fft(data_dict_before, data_dict_after, sfreq, title):
    n = len(data_dict_after)
    m = len(data_dict_before)
    # Detrend by removing the mean
    before_detrended = data_dict_before - np.mean(data_dict_before)
    after_detrended = data_dict_after - np.mean(data_dict_after)
    # Apply Hanning window
    window_after = np.hanning(n)
    window_before =np.hanning(m)
    signal_windowed_after = after_detrended * window_after
    signal_windowed_before= before_detrended * window_before
    # Compute FFT and magnitudes
    fft_result_after = np.fft.rfft(signal_windowed_after)
    fft_result_before = np.fft.rfft(signal_windowed_before)
    magnitude_before = np.abs(fft_result_before)
    magnitude_after = np.abs(fft_result_after)
    # Frequency bins
    freqs_after = np.fft.rfftfreq(n, 1/sfreq)
    freqs_before = np.fft.rfftfreq(m, 1/sfreq)
    # Plot
    plt.figure()
    plt.plot(freqs_after, magnitude_after, label="After",color="green", alpha=0.7)
    plt.plot(freqs_before, magnitude_before,label="Before" ,color="orange", alpha=0.7)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.xlim(0, 50)  # Focus on typical EEG bands (0-50 Hz)
    plt.show()

def write(file, data):
    with open(f"{file}.txt", "w") as f:
        f.write(str(data))
def printer(data_dict, channel_name):
    # Check if the channel exists in the dictionary
    if channel_name not in data_dict:
        print(f"Channel '{channel_name}' not found in the data.")
        return
    
    # Get the data for the specified channel
    channel_data = data_dict[channel_name]
    
    # Plot the data
    print(f"Plotting channel: {channel_name}")
    plt.figure()
    plt.plot(channel_data[:2500])  # Plot the first 2500 samples for better visibility
    plt.title(f"Channel: {channel_name}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
def printer_comparison(data_dict_before, data_dict_after, channel_name):
    # Check if the channel exists in both dictionaries
    if channel_name not in data_dict_before:
        print(f"Channel '{channel_name}' not found in the 'before' data.")
        return
    if channel_name not in data_dict_after:
        print(f"Channel '{channel_name}' not found in the 'after' data.")
        return

    channel_data_before = data_dict_before[channel_name]
    channel_data_after = data_dict_after[channel_name]
    plt.figure(figsize=(10, 6))
    plt.plot(channel_data_before[:31000], label="Before", color="green", alpha=0.7)
    plt.plot(channel_data_after[:31000], label="After", color="orange", alpha=0.7)
    plt.title(f"Comparison for Channel: {channel_name}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()  # Add legend to differentiate between 'before' and 'after'
    plt.grid()
    plt.show()

def compress_signal(signal, window_size=25):
    # Create the kernel for convolution (e.g., 10 points, each 0.1)
    kernel = np.ones(window_size) / window_size
    
    # Convolve the signal with the kernel
    compressed_signal = np.convolve(signal, kernel, mode='same')
    
    return compressed_signal

if __name__ == "__main__":
    input_file_before = "Subject27_1.edf"
    input_file_after = "Subject27_2.edf"
    key = "EEG F3"

    # Load data and sampling frequency
    channel_data_before, sfreq_before = load(input_file_before)
    channel_data_after, sfreq_after = load(input_file_after)

    # Ensure matching sampling frequencies
    if sfreq_before != sfreq_after:
        print("Warning: Sampling frequencies differ. Proceed with caution.")
    sfreq = sfreq_before  # Use the first file's sfreq

    # Extract signals
    signal_before = channel_data_before[key]
    signal_after = channel_data_after[key]

    # Plot FFT for each signal
    plot_fft(signal_before,signal_after, sfreq, f'FFT of {key} (Before)')
    #plot_fft(signal_after, sfreq, f'FFT of {key} (After)')
