#!/usr/bin/env python
# coding: utf-8


#!pip install matplotlib


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, medfilt
from numpy.fft import fft, fftfreq, ifft

from scipy.stats import kurtosis, skew
import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#matlab_file = './NEW/sushant/converted/log_1.mat'
# matlab_file = './NEW/soham/converted/log_1.mat'
# matlab_file = './NEW/vin/converted/log_1.mat'
# matlab_file = './Environment_CSI/converted/log1.mat'
plt.rcParams['figure.dpi'] = 275

selected_subcarrier = 0

mat_contents = None
antenna = None

def load_mat_file(matlab_file):
    mat_contents = sio.loadmat(matlab_file)
    global antenna
    antenna = mat_contents['M']

def butterworth():
    # -- APPLY BUTTERWORTH FILTER --
    filtered_stream = []
    for subcarrier in range(90):
        fs = 0.05        # Sampling Frequency
        flt_ord = 12     # Filter order number

        # Create list of CSI stream of one subcarrier
        x_axis = []
        y_axis = []
        for i in range(0, len(antenna)):
            x_axis.append(i)
            y_axis.append(antenna[i][subcarrier])

        flt_ord = 12
        b, a = butter(flt_ord, 0.05, 'lowpass', analog=False)
        output = filtfilt(b, a, y_axis)
        filtered_stream.append(output)

    return filtered_stream



# shifted_data = []
def silence_removal(stream):
    silenced_data = [] # Contains the original csi data for which each subcarrier is trimmed from best_start to best_end
    # -- SILENCE REMOVAL ---
    # Separate the CSI data into a sequence of frames
    num_packets = len(stream[selected_subcarrier])
    data_per_frame = 50    # 50 ms is the number of data(frequency) in each frame.
    total_num_frames = math.floor(num_packets/data_per_frame)
    frames = []
    for frame in range(int(total_num_frames)):
        cur_index = data_per_frame*frame
        seq = stream[selected_subcarrier][cur_index : cur_index+data_per_frame]
        frames.append(seq)

    # Calculate energy for each frame
    # Energy each frame = Average of the Squares for each frame
    energy = []
    for frame in frames:
        sq_sum = sum(map(lambda x: x*x, frame))
        e_frame = sq_sum/data_per_frame
        energy.append(e_frame)

    # Apply median filter to energy values and take their log
    filt_energy = medfilt(energy)
    filt_energy = [math.log(e) for e in filt_energy]

    # Select contiguous block of frames s_mid for which energy > mean energy

    # Get indices for which energy > mean energy
    curr_index = 0
    high_energy_indices = []
    mean_energy = sum(filt_energy)/len(filt_energy)
    for energy in filt_energy:
        if energy > mean_energy:
            high_energy_indices.append(curr_index)
        curr_index += 1

    # Split the high energy indices into contigious subsequences
    curr_index = None
    inside_arr = []
    conti_arr = []
    for index in high_energy_indices:
        if (curr_index is None):
            inside_arr.append(index)
        else:
            if (curr_index == index-1):
                inside_arr.append(index)
            else:
                conti_arr.append(inside_arr)
                inside_arr = []
                inside_arr.append(index)

        curr_index = index

    if inside_arr:
        conti_arr.append(inside_arr)

    # Select the longest contigious subsequence
    s_mid = max(conti_arr, key=len)

    # Get the corresponding CSI data frames h_mid for s_mid
    h_mid = []
    for index in s_mid:
        h_mid.append(frames[index])


    # Calculate the midpoint of the region h_mid which has maximum
    # deviations from the average in h_mid
    high_energy_csi = [item for sublist in h_mid for item in sublist]
    avg_energy = sum(high_energy_csi)/len(high_energy_csi)

    max_dev = 0
    max_dev_index = 0
    for index, energy in enumerate(high_energy_csi):
        deviation = abs(avg_energy - energy)
        if deviation > max_dev:
            max_dev = deviation
            max_dev_index = index

    m = s_mid[0] * data_per_frame + max_dev_index

    # Set start_point to m−T/2, where T is total duration
    start_point = math.floor(m - (num_packets)/2)
    start_point = max(start_point, 0)

    # Set end_point to m+T/2, where T is total duration
    end_point = math.floor(m + (num_packets)/2)
    end_point = min(end_point, num_packets)

    # print("Midpoint = ", m)
    # print("Numpackets = ", num_packets)
    # get a 4 second window from mid
    # if num_packets >= 4000:
    #     start_point = m - 2000
    #     end_point = m + 2000
    #     carry = 0

    #     if start_point < 0:
    #         carry = abs(start_point)
    #         start_point = 0
    #         end_point += carry

    #     if end_point >= num_packets:
    #         carry = end_point - num_packets
    #         end_point = num_packets
    #         if (start_point - carry) >= 0:
    #             start_point -= carry
    # else:
    #     start_point = 0
    #     end_point = num_packets

    # print("start = ", start_point)
    # print("end = ", end_point)

    # Get original data with silence removed
    # trimmed_output = stream[selected_subcarrier][start_point:end_point]

    for untrimmed in stream:
        new_csi = untrimmed[start_point:end_point]
        silenced_data.append(new_csi)

    return silenced_data

    # # Plot the silenced CSI data
    # for silenced in silenced_data:
    #     plt.plot(silenced)
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.title('Filtered CSI stream (Ord 12) - Silence removed')

    # Shift data by mean
    # silenced_csi_signal = silenced_data[selected_subcarrier]
    # mean_silenced = sum(silenced_csi_signal)/len(silenced_csi_signal)
    # shifted_data = [i - mean_silenced for i in silenced_csi_signal]

    # plt.axhline(y=0, color='r', linestyle='-')
    # plt.plot(shifted_data)



def perform_fft():
    # Perform FFT
    n = len(shifted_data)
    timestep = 1
    freq = fftfreq(n, d=timestep)
    mask = freq > 0
    xmin = 0.002; xmax = 0.008
    ymin = 0; ymax = 1

    frequency_data = fft(shifted_data)
    print(frequency_data)
    fft_theo = 2.0*np.abs(frequency_data/n)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.plot(freq[mask], fft_theo[mask])

    # plt.plot(frequency_data, fft_theo)

    # print(len(freq[mask]))
    # print(n)

    # plt.ylim(bottom=0.002, top=0.008)
    # plt.plot([i for i in range(n)], freq)

    # print(fft_theo[mask])
    # print(freq)
    separated_freq = [f for f in freq if (f > 0.002 and f < 0.008)]
    actual_val = ifft(separated_freq)

def extract_features():
    silenced_data = []
    for subcarrier in range(90):
        # Create list of CSI stream of one subcarrier
        ms_data = []
        for i in range(0, len(antenna)):
            ms_data.append(antenna[i][subcarrier])

        silenced_data.append(ms_data)

# Feature extraction - all subcarriers
    features = []
    for subcarrier in range(len(silenced_data)):
        num_packets = len(silenced_data[subcarrier])
        data_per_window = 100
        total_num_windows = math.floor(num_packets/data_per_window)

        # Split data into windows of 100 packets / 0.1 seconds
        windows = []
        for window in range(int(total_num_windows)):
            cur_index = data_per_window*window
            seq = silenced_data[subcarrier][cur_index : cur_index+data_per_window]

            windows.append(seq)

        # For each window, gather its features
        for each_window in windows:
            # time domain features

            weight_each_window = []

            mean = sum(each_window) / len(each_window)
            max_val = max(each_window)
            min_val = min(each_window)
            skewness = skew(each_window)
            kurtosis_val = kurtosis(each_window)
            variance = np.var(each_window)

            weight_each_window.append(mean)
            weight_each_window.append(max_val)
            weight_each_window.append(min_val)
            weight_each_window.append(skewness)
            weight_each_window.append(kurtosis_val)
            weight_each_window.append(variance)

            features.append(weight_each_window)
    return features

def export_sushant_data(features):
# Export Sushant data
    sushant_data = {
        'features': features,
    }

    with open('sushant_data.json', 'w') as outfile:
        json.dump(sushant_data, outfile, indent=4)

def export_soham_data(features):
    # Export Soham data
    soham_data = {
        'features': features,
    }

    with open('soham_data.json', 'w') as outfile:
        json.dump(soham_data, outfile, indent=4)

def export_vintony_data(features):
# Export Vintony data
    vintony_data = {
        'features': features,

    }

    with open('vintony_data.json', 'w') as outfile:
        json.dump(vintony_data, outfile, indent=4)


# # Export Environment data
# environment_data = {
#     'feature_names': [
#         'mean',
#         'max_val',
#         'min_val',
#         'skewness',
#         'kurtosis_val',
#         'variance'
#     ],
#     'features': features,
#     'target_name': 'Environment',
#     'target': 3
# }

# with open('environment_data.json', 'w') as outfile:
#     json.dump(environment_data, outfile, indent=4)


def get_data():

    # Preprocess for building classifier
    with open('sushant_data.json') as json_file:
        sushant = json.load(json_file)

    with open('soham_data.json') as json_file:
        soham = json.load(json_file)

    with open('vintony_data.json') as json_file:
        vintony = json.load(json_file)

    data_length = len(sushant['features']) + len(soham['features']) + len(vintony['features'])
    csi_data ={
        'feature_names': [
            'mean',
            'max_val',
            'min_val',
            'skewness',
            'kurtosis_val',
            'variance'
        ],
        'features': np.array(sushant['features'] + soham['features'] + vintony['features']),
        'target_names': ['Sushant', 'Soham', 'Vintony'],
        'target': np.empty(data_length)
    }
    csi_data['target'][0:len(sushant['features'])] = 0
    csi_data['target'][len(sushant['features']): len(soham['features'])+len(sushant['features'])] = 1
    csi_data['target'][len(soham['features']) + len(sushant['features']): data_length] = 2

    print("Number of 2s: ", len(vintony['features']))
    print("Number of 1s: ", len(soham['features']))
    print("Number of 0s: ", len(sushant['features']))

    data = pd.DataFrame({
            'mean': csi_data['features'][:,0],
            'max_val': csi_data['features'][:,1],
            'min_val': csi_data['features'][:,2],
            'skewness': csi_data['features'][:,3],
            'kurtosis_val': csi_data['features'][:,4],
            'variance': csi_data['features'][:,5],
            'person': csi_data['target']
    })

    data = shuffle(data)

    # features
    X = data[['mean', 'max_val', 'min_val', 'skewness', 'kurtosis_val', 'variance']]
    # labels
    y = data['person']

    # Split dataset into training set and test set
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
    return train_test_split(X, y, test_size=0.3)
    # return X, y