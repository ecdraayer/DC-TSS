import numpy as np
import pandas as pd
import random
import os
import csv
import IPython.display as ipd
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition
import librosa
import librosa.display
import math
import mne
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from numpy.linalg import norm

def smooth(weights, arr):
    return np.convolve(weights/weights.sum(), arr, mode='same')

def get_dataloader(data, batch_size, num_workers=0, data_transforms=None):
    if data_transforms is None:
        data_transforms = transforms.ToTensor()
        
    data_tensor = torch.from_numpy(data)
    
    data_loader = DataLoader(dataset=data_tensor, 
                             batch_size=batch_size, 
                             num_workers=num_workers)
    
    return data_loader

def get_changepoints(real_peaks, window_length, overlap):
    change_points = []
    diff = window_length - int(overlap*window_length)
    for peak in real_peaks:
        change_point = window_length
        change_point += diff*peak
        change_point -= window_length//2
        change_points.append(int(change_point))
        
    return change_points
def manhattan_distance(x, y):
    """
    Calculate the Manhattan distance between two vectors x and y.

    Args:
        x (numpy.ndarray): The first vector.
        y (numpy.ndarray): The second vector.

    Returns:
        The Manhattan distance between x and y.
    """
    return np.sum(np.abs(x - y))

def get_label_score(data, window_length):
    assert window_length%2 == 0
       
    distances = []
    
    window_length = window_length//2
    for i in range(len(data) - 2*window_length):
        window1 = np.bincount(data[i:i+window_length])
        window1 = np.pad(window1, (0,len(np.unique(data)) - len(window1)), 'constant')
        #window1 = data[i:i+window_length]
        
        window2 = np.bincount(data[i+window_length: i+2*window_length])
        window2 = np.pad(window2, (0,len(np.unique(data)) - len(window2)), 'constant')
        #window2 = data[i+window_length: i+2*window_length]
        #distance = np.dot(window1,window2)/(norm(window1)*norm(window2))
        distance = manhattan_distance(window1, window2)
        distances.append(distance)
    
    distances = np.asarray(distances)
    return distances

def relative_change_point_distance(cps_true, cps_pred, ts_len):
    '''
    Calculates the relative CP distance between ground truth and predicted change points.
    Parameters
    -----------
    :param cps_true: an array of true change point positions
    :param cps_pred: an array of predicted change point positions
    :param ts_len: the length of the associated time series
    :return: relative distance between cps_true and cps_pred considering ts_len
    >>> score = relative_change_point_distance(cps, found_cps, ts.shape[0])
    '''
    assert len(cps_true) == len(cps_pred), "true/predicted cps must have the same length."
    differences = 0

    for cp_pred in cps_pred:
        distances = paired_euclidean_distances(
            np.array([cp_pred]*len(cps_true)).reshape(-1,1),
            cps_true.reshape(-1,1)
        )
        cp_true_idx = np.argmin(distances, axis=0)
        cp_true = cps_true[cp_true_idx]
        differences += np.abs(cp_pred-cp_true)

    return np.round(differences / (len(cps_true) * ts_len), 6)


def _true_positives(T, X, margin=5):
    '''
    Compute true positives without double counting
    Author: G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    Examples
    -----------
    >>> _true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> _true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> _true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> _true_positives(set(), {1, 2, 3})
    set()
    >>> _true_positives({1, 2, 3}, set())
    set()
    '''
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(ground_truth, predictions, margin=5, alpha=0.5, return_PR=False):
    '''
    Compute the F-measure based on human annotations. Remember that all CP locations are 0-based!
    Author: G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    Parameters
    -----------
    :param annotations: dict from user_id to iterable of CP locations
    :param predictions: iterable of predicted CP locations
    :param alpha: value for the F-measure, alpha=0.5 gives the F1-measure
    :return: whether to return precision and recall too
    Examples
    -----------
    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8
    '''
    annotations =	{'1':ground_truth}
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(_true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: _true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def _overlap(A, B):
    '''
    Return the overlap (i.e. Jaccard index) of two sets
    Author: G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    Examples
    -----------
    >>> _overlap({1, 2, 3}, set())
    0.0
    >>> _overlap({1, 2, 3}, {2, 5})
    0.25
    >>> _overlap(set(), {1, 2, 3})
    0.0
    >>> _overlap({1, 2, 3}, {1, 2, 3})
    1.0
    '''
    return len(A.intersection(B)) / len(A.union(B))


def _partition_from_cps(locations, n_obs):
    '''
    Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.
    Author: G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    Examples
    -----------
    >>> _partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> _partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> _partition_from_cps([1,2,7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> _partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    '''
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def _cover_single(Sprime, S):
    '''
    Compute the covering of a segmentation S by a segmentation Sprime.
    This follows equation (8) in Arbaleaz, 2010.
    Author: G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    Examples
    -----------
    >>> _cover_single([{1, 2, 3}, {4, 5}, {6}], [{1, 2, 3}, {4, 5, 6}])
    0.8333333333333334
    >>> _cover_single([{1, 2, 3, 4}, {5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.6666666666666666
    >>> _cover_single([{1, 2}, {3, 4}, {5, 6}], [{1, 2, 3}, {4, 5, 6}])
    0.6666666666666666
    >>> _cover_single([{1, 2, 3, 4, 5, 6}], [{1}, {2}, {3}, {4, 5, 6}])
    0.3333333333333333
    '''
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(_overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(ground_truth, predictions, n_obs):
    '''
    Compute the average segmentation covering against the human annotations.
    Author: G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    Parameters
    -----------
    @param annotations: dict from user_id to iterable of CP locations
    @param predictions: iterable of predicted Cp locations
    @param n_obs: number of observations in the series
    Examples
    -----------
    >>> covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45)
    0.7962962962962963
    >>> covering({1: [], 2: [10], 3: [40]}, [10], 45)
    0.7954144620811286
    >>> covering({1: [], 2: [10], 3: [40]}, [], 45)
    0.8189300411522634
    '''
    annotations =	{'1':ground_truth}
    Ak = {
        k + 1: _partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = _partition_from_cps(predictions, n_obs)

    Cs = [_cover_single(pX, Ak[k]) for k in Ak]
    return sum(Cs) / len(Cs)

def get_daily_sports_timeseries(path, person, activities, durations, preprocess=None):
    directory = path
    daily_sports_data = []
    for filename in os.listdir(directory):
        sub_directory = os.path.join(directory, filename)
        for file in os.listdir(sub_directory):
            if file == 'p{}'.format(person):
                activity_data = []
                for datum in os.listdir(os.path.join(sub_directory, file)):
                    # with open(sub_directory+'\\p{}\\'.format(person)+datum, mode='r') as csv_file:
                    data = np.genfromtxt(sub_directory + '\\p{}\\'.format(person) + datum, delimiter=',')
                    # for row in data:
                    activity_data.append(data)

                daily_sports_data.append(activity_data)

    daily_sports_data = np.asarray(daily_sports_data)

    time_series, labels = construct_daily_sports_activity(daily_sports_data, activities, durations)
    
    if ( preprocess == 0 ):
        print((np.max(time_series, axis=1) - np.min(time_series, axis=1)).shape)
        print((np.max(time_series, axis=1) - np.min(time_series, axis=1)))
        time_series = ((time_series.T - np.min(time_series, axis=1)) / (np.max(time_series, axis=1) - np.min(time_series, axis=1))).T
    elif ( preprocess == 1 ):
        time_series = ((time_series.T - np.mean(time_series, axis=1)) / np.std(time_series, axis=1)).T
    else:
        None
        
    return time_series, labels


def construct_daily_sports_activity(daily_sports_data, activities, durations):
    time_series = []
    labels = []

    for i, activity in enumerate(activities):
        clips = np.random.randint(low=0, high=60, size=durations[i])
        for n in range(durations[i] * 125):
            labels.append(activity)
        for clip in clips:
            time_series.append(daily_sports_data[activity][clip])

    time_series = [item for sublist in time_series for item in sublist]

    time_series = np.asarray(time_series)
    labels = np.asarray(labels)
        
    return time_series.T, labels

def ffill_loop(arr, fill=0):
    mask = np.isnan(arr[0])
    arr[0][mask] = fill
    for i in range(1, len(arr)):
        mask = np.isnan(arr[i])
        arr[i][mask] = arr[i - 1][mask]
    return arr



def ffill_loop(arr, fill=0):
    mask = np.isnan(arr[0])
    arr[0][mask] = fill
    for i in range(1, len(arr)):
        mask = np.isnan(arr[i])
        arr[i][mask] = arr[i - 1][mask]
    return arr



def get_pamap_dataset(path, person, preprocess=None):
    pamap_data = np.genfromtxt(path+"\\subject{}.dat".format(person))
    #fill_loop(pamap_data)
    #print(pamap_data.shape)
    #time_series = pamap_data[:,2:]
    #time_series = time_series.T
    #print(time_series.shape)
    indices= np.array([14, 15, 16, 17, 31, 32, 33, 34, 48, 49, 50, 51])
    #pamap_data = np.delete(time_series, indices, axis=0)
    
    df = pd.DataFrame(pamap_data)
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    #ffill_loop(pamap_data)
    pamap_data = df.to_numpy()
    labels = pamap_data[:,1]
    time_series = pamap_data[:,2:]
    time_series = time_series.T
    #indices=np.where(np.std(time_series,axis=1)==0)
    time_series = np.delete(time_series, indices, axis=0)
    if ( preprocess == 0 ):
        time_series = ((time_series.T - np.min(time_series, axis=1)) / (np.max(time_series, axis=1) - np.min(time_series, axis=1))).T
    elif ( preprocess == 1 ):
        time_series = ((time_series.T - np.mean(time_series, axis=1)) / np.std(time_series, axis=1)).T
    else:
        None
    return time_series, labels


def get_music_dataset(path, selected_songs, preprocess=1):
    data = []
    labels = []
    labels1 = []

    AUDIO_DIR = path
    tracks = audioUtils.load('data/fma_metadata/tracks.csv')

    music_data = []
    c_song_label = 0
    h_length = 128


    for song in selected_songs:
        print("song:",song)
        filename = audioUtils.get_audio_path(AUDIO_DIR, song)

        x, sr = librosa.load(filename, sr=None, mono=True)
        music_data.append(x)
        #print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

        c_label = tracks.loc[song]['track','genre_top']

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=h_length))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        log_mel = librosa.amplitude_to_db(mel)
        #scaler = skl.preprocessing.StandardScaler()
        #$fcc = scaler.fit_transform(mfcc)
#         librosa.display.specshow(log_mel, sr=sr, hop_length=h_length, x_axis='time', y_axis='mel');
        
        for i in range(len(log_mel[0])):
            labels1.append(c_song_label)
            labels.append(c_label)
        c_song_label += 1

    music_data = np.hstack(music_data)

    stft = np.abs(librosa.stft(music_data, n_fft=2048, hop_length=h_length))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)
    log_mel = (log_mel - np.mean(log_mel)) / np.std(log_mel)
    #scaler = skl.preprocessing.StandardScaler()
    #log_mel = scaler.fit_transform(log_mel.T).T
    #librosa.display.specshow(log_mel, sr=sr, hop_length=h_length, x_axis='time', y_axis='mel')
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    scaler = skl.preprocessing.StandardScaler()
    mfcc = scaler.fit_transform(mfcc)
    #librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    #print(mfcc.shape)
    #mfcc = mfcc.T

    #mfcc = mfcc.T
    #librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    #time_series = log_mel
    #print(scaler.mean_.shape)
    time_series = np.concatenate((log_mel, mfcc), axis=0)
    
    #time_series = scaler.fit_transform(time_series)
    #print(np.mean(time_series, axis=1).shape)
#     if ( preprocess == 0 ):
#         time_series = ((time_series.T - np.min(time_series, axis=1)) / (np.max(time_series, axis=1) - np.min(time_series, axis=1))).T
#     elif ( preprocess == 1 ):
#         time_series = ((time_series.T - np.mean(time_series, axis=1)) / np.std(time_series, axis=1)).T
#     else:
#         time_series = time_series.T
    activity_names = np.unique(labels, return_inverse=True)[0]
    labels = np.unique(labels, return_inverse=True)[1]
    labels1 = np.array(labels1)
    
    return time_series, labels1, activity_names, labels

def get_eeg_dataset(path, preprocess):
    activities = [0,1,2,3,4,5,2,3,4,5,2,3,4,5]
    labels = []
    time_series = []
    for root, dirs, files in os.walk(path):
        print(root)
        for i, file in enumerate(files):
            data = mne.io.read_raw_edf(root+"\\"+file)
            raw_data = data.get_data()
            for j in range(len(raw_data[0])):
                labels.append(activities[i])
                #labels.append(activities[i])
            #time_series.append(raw_data)
            time_series.append(raw_data)
    time_series = np.hstack(time_series)
    if ( preprocess == 0 ):
        time_series = ((time_series.T - np.min(time_series, axis=1)) / (np.max(time_series, axis=1) - np.min(time_series, axis=1))).T
    elif ( preprocess == 1 ):
        time_series = ((time_series.T - np.mean(time_series, axis=1)) / np.std(time_series, axis=1)).T
    else:
        time_series = time_series.T
    
    labels = np.asarray(labels)
    
    return time_series, labels

def breakpoint_distance(embedded_features):
    breakpoint_distances = []
    for i,e_feature in enumerate(embedded_features[1:]):
        numerator = np.linalg.norm(e_feature - embedded_features[i] )
        denominator = math.sqrt(np.linalg.norm(embedded_features[i]) * np.linalg.norm(e_feature))
        breakpoint_distances.append(numerator / denominator)
        
    breakpoint_distances = np.array(breakpoint_distances)
    return breakpoint_distances

def combine_predictions(predictions, error):
    reduced_predictions = []
    last = np.unique(np.sort(predictions))[0]
    true_prediction = last
    count = 1
    for pred in np.unique(np.sort(predictions))[1:]:
        if pred - last < error:
            true_prediction += pred
            count+=1
            last = pred 
        else:
            reduced_predictions.append(true_prediction//count)
            true_prediction = pred
            last = pred
            count = 1
    reduced_predictions.append(true_prediction//count)
    reduced_predictions = np.array(reduced_predictions).astype(int)
    
    return reduced_predictions
