# coding: utf-8
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, filtfilt, medfilt
from numpy.fft import fft, fftfreq, ifft
from scipy.stats import kurtosis, skew, mode
from joblib import load
import pandas as pd

import torch

from Wifi_neural import FeedForward

def get_silence_remove(matlab_file):
    original = [] # Dataset with csi streams, filtered
    selected_subcarrier = 0

    mat_contents = sio.loadmat(matlab_file)
    antenna = mat_contents['M']

    # -- APPLY BUTTERWORTH FILTER --
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
        original.append(output)


    # -- SILENCE REMOVAL ---
    # Separate the CSI data into a sequence of frames
    num_packets = len(original[selected_subcarrier])
    data_per_frame = 50    # 50 ms is the number of data(frequency) in each frame.
    total_num_frames = math.floor(num_packets/data_per_frame)
    frames = []
    for frame in range(int(total_num_frames)):
        cur_index = data_per_frame*frame
        seq = original[selected_subcarrier][cur_index : cur_index+data_per_frame]
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

    silenced_data = [] # Contains the original csi data for which each subcarrier is trimmed from best_start to best_end
    for untrimmed in original:
        new_csi = untrimmed[start_point:end_point]
        silenced_data.append(new_csi)

    return silenced_data

def extract_features(silenced_data):
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

if __name__ == "__main__":
    matlab_file = './NEW/soham/converted/log_2.mat'
    silenced_data = get_silence_remove(matlab_file)
    features = extract_features(silenced_data)
    # print(features)
    # print(np.array(features))

    features = np.array(features)
    data = pd.DataFrame({
        'mean': features[:,0],
        'max_val': features[:,1],
        'min_val': features[:,2],
        'skewness': features[:,3],
        'kurtosis_val': features[:,4],
        'variance': features[:,5],
    })

    # features
    X = data[['mean', 'max_val', 'min_val', 'skewness', 'kurtosis_val', 'variance']]
    X = torch.tensor(X.values)

    model = torch.load('./model.pth')
    #model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    last_num = math.floor(X.shape[0]/10)*10
    X = X[:last_num]
    X = X.view(X.shape[0], 1, X.shape[1])
    X = X.view(math.floor(X.shape[0]/10), 10, 6)

    X = X[200]
    log_ps = model(X.float())

    predictions = []
    for idx, i in enumerate(log_ps):
        print(i)
        predictions.append(torch.argmax(i))

    print(predictions)
    #load trained model
    # clf = load("CSI_MODEL.joblib")
    # output = clf.predict(X)
    # print(output)
    # print(mode(output))