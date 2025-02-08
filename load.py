import mne
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=sys.maxsize)

def load(file):
    data = mne.io.read_raw_edf(file, preload=True)
    raw_data = data.get_data()
    channels = data.ch_names
    sfreq = data.info['sfreq']
    channel_data_dict = {channels[i]: raw_data[i] for i in range(len(channels))}
    print(f"Channels loaded: {channels}, Sampling frequency: {sfreq} Hz")
    return channel_data_dict, sfreq

def compute_fft(signal_before, signal_after, sfreq):
    min_len = min(len(signal_before), len(signal_after))
    before_trimmed = signal_before[:min_len]
    after_trimmed = signal_after[:min_len]
    
    before_detrended = before_trimmed - np.mean(before_trimmed)
    after_detrended = after_trimmed - np.mean(after_trimmed)
    
    window = np.hanning(min_len)
    fft_before = np.fft.rfft(before_detrended * window) / np.sum(window)
    fft_after = np.fft.rfft(after_detrended * window) / np.sum(window)
    
    return (np.fft.rfftfreq(min_len, 1/sfreq), 
            np.abs(fft_before), 
            np.abs(fft_after))

def plot_all_comparisons(channel_data_before, channel_data_after, sfreq):
    common_channels = sorted(set(channel_data_before.keys()) & set(channel_data_after.keys()))
    
    if not common_channels:
        print("No common channels found between before/after data!")
        return
    
    n_channels = len(common_channels)
    n_cols = 2
    n_rows = math.ceil(n_channels / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axs = axs.flatten() if n_channels > 1 else [axs]
    
    for idx, channel in enumerate(common_channels):
        freqs, mag_before, mag_after = compute_fft(
            channel_data_before[channel],
            channel_data_after[channel],
            sfreq
        )
        
        # Plot with distinct styles
        axs[idx].plot(freqs, mag_before, color='blue', linestyle='-', linewidth=1.5, alpha=0.8, label='Before')
        axs[idx].plot(freqs, mag_after, color='red', linestyle='-.', linewidth=1.5, alpha=0.8, label='After')
        
        axs[idx].set_title(f'{channel} FFT Comparison', fontsize=12)
        axs[idx].set_xlabel('Frequency (Hz)', fontsize=10)
        axs[idx].set_ylabel('Normalized Magnitude', fontsize=10)
        axs[idx].grid(True, linestyle=':', alpha=0.7)
        axs[idx].set_xlim(0, 50)
        axs[idx].legend(fontsize=9)
        axs[idx].tick_params(axis='both', which='major', labelsize=8)
    
    # Hide empty subplots
    for j in range(n_channels, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plot_fft(data_dict_before, data_dict_after, sfreq, title):
    # Trim signals to the same length (use the shorter one)
    min_len = min(len(data_dict_before), len(data_dict_after))
    before_trimmed = data_dict_before[:min_len]
    after_trimmed = data_dict_after[:min_len]
    
    # Detrend
    before_detrended = before_trimmed - np.mean(before_trimmed)
    after_detrended = after_trimmed - np.mean(after_trimmed)
    
    # Apply Hanning window
    window = np.hanning(min_len)
    signal_windowed_before = before_detrended * window
    signal_windowed_after = after_detrended * window
    
    # Compute FFT and normalize by window sum
    fft_before = np.fft.rfft(signal_windowed_before) / np.sum(window)
    fft_after = np.fft.rfft(signal_windowed_after) / np.sum(window)
    
    magnitude_before = np.abs(fft_before)
    magnitude_after = np.abs(fft_after)
    
    # Frequency bins
    freqs = np.fft.rfftfreq(min_len, 1/sfreq)
    
    # Plot
    plt.figure()
    plt.plot(freqs, magnitude_after, label="After", color="green", alpha=0.7)
    plt.plot(freqs, magnitude_before, label="Before", color="orange", alpha=0.7)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.legend()
    plt.grid()
    plt.xlim(0, 50)
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
    input_file_before = "Subject23_1.edf"     #18 dobar
    input_file_after =  "Subject23_2.edf"    #18
    key = "EEG A2-A1"

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
    plot_all_comparisons(channel_data_before, channel_data_after, sfreq)

    #plot_fft(signal_before,signal_after, sfreq, f'FFT of {key} (Before)')
    #plot_fft(signal_after, sfreq, f'FFT of {key} (After)')
