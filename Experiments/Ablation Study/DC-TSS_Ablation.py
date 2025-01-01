import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
import csv
from utils import *
from TS_DEC import *
from scipy import stats as st

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.autograd import Variable
import math
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_distance
from typing import Optional, List
from numpy.linalg import norm
from scipy.io import arff
import pandas as pd
import mne
from scipy.signal import find_peaks
import time
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("data")
args = parser.parse_args()

string = args.data
print(string)
if string  == 'PAMAP2':
    print("processing PAMAP2")
    batch_size = 1000
    epochs1 = 1200
    epochs2 = 8000
    n_clusters = 70
    lr=0.08
    pre_lr = 1.5e-4
    momentum=0.9
    update_interval = 1

    window_length = 150
    overlap_percent = 0.3

    window_length2 = 250
    margin = 100*30
    window_length2 = 50
    blackman_window = 70
    k_sizes = [10,6,6]
    strides = [3,3,3]
    prominence = 0.05
    height = 0.5
    n_clusters_list = 70

    time_series = np.loadtxt("./data/PAMAP2.csv", delimiter=",")
    labels = np.loadtxt("./data/PAMAP2_labels.csv", delimiter=",")
elif string == 'EEG':
    k_sizes = [10,8,5]
    strides = [3,3,2]
    batch_size = 1000
    epochs1 = 1200
    epochs2 = 8000
    n_clusters = 60
    lr=0.03
    momentum=0.9
    pre_lr = 1.5e-4
    window_length = 150
    overlap_percent = 0.5
    window_length2 = 250
    blackman_window = 100
    print("processing EEG")
    time_series = np.loadtxt("./data/EEG.csv", delimiter=",")
    labels = np.loadtxt("./data/EEG_labels.csv", delimiter=",")
    margin = 160*25
    prominence = 0.1
    height = 0.4
    n_clusters_list = 60

elif string == 'MUSIC':
    k_sizes = [10,8,6]
    strides = [3,3,3]
    batch_size = 200
    epochs1 = 1200
    epochs2 = 8000
    update_interval = 1
    n_clusters = 40
    lr=0.05
    momentum=0.1
    margin = 1800
    window_length = 150
    overlap_percent = 0.50
    window_length2 = 150  
    blackman_window = 100
    prominence = 0.05
    height = 0.1
    n_clusters_list = 40
    print("processing MUSIC")
    time_series = np.loadtxt("./data/Music_Analysis.csv", delimiter=",")
    labels = np.loadtxt("./data/Music_Analysis_labels.csv", delimiter=",")

else:
    print("processing SPORTS")
    time_series = np.loadtxt("./data/Sports_Activity.csv", delimiter=",")
    k_sizes = [6,4,4]
    strides = [2,2,2]
    batch_size = 1000
    epochs1 = 1200
    epochs2 = 8000
    update_interval = 1
    n_clusters = 80
    n_clusters_list = 80

    pre_lr = 1.5e-4
    lr=0.01
    momentum=0.9
    margin = 225
    window_length = 25
    overlap_percent = 0.20
    window_length2 = 200
    blackman_window = 50
    labels = np.loadtxt("./data/Sports_Activity_labels.csv", delimiter=",")
    margin = 200
    prominence = 0.1
    height = 0.5

ground_truth = np.where(labels[:-1] != labels[1:])[0]


start = 0
subsequences = []
subsequence_labels = []

while start+window_length < len(time_series[0]):
    subsequence_labels.append(st.mode(labels[start:start+window_length])[0][0])    
    subsequence = time_series[:,start:start+window_length]
    start = start+window_length - int(overlap_percent*window_length)
    subsequences.append(subsequence)

subsequences = np.asarray(subsequences)
subsequence_labels = np.asarray(subsequence_labels)
print(subsequences.shape)
print(len(subsequence_labels))

def loss_fn(recon_z, x):
    BCE = F.mse_loss(recon_z, x)
    return BCE

def get_dataloader(data, batch_size, num_workers=0, data_transforms=None):
    if data_transforms is None:
        data_transforms = transforms.ToTensor()
        
    data_tensor = torch.from_numpy(data)
    
    data_loader = DataLoader(dataset=data_tensor, 
                             batch_size=batch_size, 
                             num_workers=num_workers)
    
    return data_loader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_loader = get_dataloader(data=subsequences,
                          batch_size=batch_size,
                          num_workers=1,
                          data_transforms=None)

torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

input_shape = subsequences.shape
deep_cluster_model = DEC(n_clusters = n_clusters, input_shape=input_shape,k_sizes=k_sizes,strides=strides)
deep_cluster_model.to(device)

start = 0
subsequences = []
subsequence_labels = []

while start+window_length < len(time_series[0]):
    subsequence_labels.append(st.mode(labels[start:start+window_length])[0][0])    
    subsequence = time_series[:,start:start+window_length]
    start = start+window_length - int(overlap_percent*window_length)
    subsequences.append(subsequence)

subsequences = np.asarray(subsequences)
subsequence_labels = np.asarray(subsequence_labels)
print(subsequences.shape)
print(len(subsequence_labels))


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_loader = get_dataloader(data=subsequences,
                          batch_size=batch_size,
                          num_workers=1,
                          data_transforms=None)
torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

input_shape = subsequences.shape
deep_cluster_model = DEC(n_clusters = n_clusters, input_shape=input_shape,k_sizes=k_sizes,strides=strides)
deep_cluster_model.to(device)
pretraining(model=deep_cluster_model, dbgenerator=data_loader, batch_size=batch_size, epochs=epochs1)

clustering_output = []
for ts in data_loader:
    recon = deep_cluster_model.AE.encode((ts.float().to(device)))
    clustering_output.append( recon.cpu().detach().numpy() ) 


print(len(clustering_output))
clustering_output = [item for sublist in clustering_output for item in sublist]
print(len(clustering_output))
clustering_output = np.asarray(clustering_output)
print(clustering_output.shape)
clustering_output_f = []
for i,co in enumerate(clustering_output):
    clustering_output[i].flatten()
    clustering_output_f.append(clustering_output[i].flatten())
clustering_output_f = np.asarray(clustering_output_f)
print(clustering_output_f.shape)

cluster_assignments = AgglomerativeClustering(n_clusters = n_clusters).fit(clustering_output_f)
cluster_assignments = cluster_assignments.labels_    

cluster_assignmentss = np.concatenate((cluster_assignments[:window_length2//2], cluster_assignments), axis=None)
cluster_assignmentss = np.concatenate((cluster_assignmentss, cluster_assignments[-window_length2//2:]), axis=None)
similarities = get_label_score(cluster_assignmentss, window_length2)
data = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

data = smooth(np.blackman(blackman_window), data)

peaks, peak_data = find_peaks(data, height=height,distance=window_length2//3,prominence=prominence)
real_peaks = peaks
predictions = get_changepoints(real_peaks, window_length, overlap_percent)

print('covering score:',covering(ground_truth, predictions, len(labels)))
print('f_measure score:',f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=True))

print("========================================================")
