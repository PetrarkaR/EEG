import mne
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=sys.maxsize)

def load(file):
    # Load EDF file and get raw data
    data = mne.io.read_raw_edf(file, preload=True)
    raw_data = data.get_data()  # Get raw signal as a numpy array
    channels = data.ch_names   # Get channel names
    raw_data = np.array(raw_data)
    print("Channels loaded:", channels)
    
    # Map channel names to their respective data
    channel_data_dict = {channels[i]: raw_data[i] for i in range(len(channels))}
    
    return channel_data_dict

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
    # Example usage
    input_file_before = "Subject27_1.edf"  # Replace with your actual file path
    input_file_after = "Subject27_2.edf"  # Replace with your actual file path

    output_file = "output_data"     # Output file name without extension
    start_time = time.monotonic()
    key = "EEG P4"
    # Load data and create channel-to-data dictionary
    channel_data_dict_before = load(input_file_before)
    channel_data_dict_after = load(input_file_after)
    signal_before = channel_data_dict_before[key]
    signal_after = channel_data_dict_after[key]
    compressed_after=compress_signal(signal_after)
    compressed_before=compress_signal(signal_before)
    compressed_before=compressed_before[0:30999]
    signal_before=signal_before[0:30999]
    #print(len(channel_data_dict_after[key]))
    #printer(channel_data_dict_before,"EEG Fp1")
    #printer(channel_data_dict_after,"EEG Fp1")
    # Example: Plot data for each channel
    #printer_comparison(channel_data_dict_before, channel_data_dict_after, "EEG A2-A1")
    #printer_comparison(compressed_after, compressed_before, "EEG Fp2")
    avg_after=np.average(compressed_after)
    avg_before=np.average(compressed_before)
    if(avg_after>avg_before):
        print(f"after is more active with {avg_after}V rather than {avg_before}V")
        print(f"peaks are after:{np.max(compressed_after)}V and before:{np.max(compressed_before)}V")
    else:
        print(f"before is more active with {avg_before}V rather than {avg_after}V")
        print(f"peaks are after:{np.max(compressed_after)}V and before:{np.max(compressed_before)}V")
        
    plt.figure(figsize=(10, 6))
    plt.plot(signal_before, label="Before (Compressed)", color="green", alpha=0.9)
    plt.plot(signal_after[0:30999], label="After (Compressed)", color="orange", alpha=0.7)
    plt.title(f"Compressed Comparison for Channel: {key}")
    plt.xlabel("Samples (Compressed)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


    end_time = time.monotonic()
    print(f"Execution time: {end_time - start_time} seconds")
