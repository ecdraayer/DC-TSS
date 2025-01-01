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
#from sklearn.cluster import KMeans
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
    batch_size = 1024
    epochs1 = 1200
    epochs2 = 8000
    n_clusters = 60
    lr=0.1
    pre_lr = 1.5e-4
    momentum=0.9
    update_interval = 1

    window_length = 150
    overlap_percent = 0.3

    window_length2 = 250
    margin = 100*30
    window_length2 = 50
    blackman_window = 70
    k_sizes = [12,7,4]
    strides = [3,3,3]
    prominence = 0.08
    height = 0.5

    time_series = np.loadtxt("./data/PAMAP2.csv", delimiter=",")
    labels = np.loadtxt("./data/PAMAP2_labels.csv", delimiter=",")
    n_clusters_list = [10,40,70,100,130,160]

elif string == 'EEG':
    k_sizes = [10,8,5]
    strides = [3,3,2]
    batch_size = 1024
    epochs1 = 1200
    epochs2 = 8000
    lr=0.05
    momentum=0.2
    pre_lr = 1.5e-4
    n_clusters = 60
    window_length = 150
    overlap_percent = 0.5
    window_length2 = 250
    blackman_window = 50
    print("processing EEG")
    time_series = np.loadtxt("./data/EEG.csv", delimiter=",")
    labels = np.loadtxt("./data/EEG_labels.csv", delimiter=",")
    margin = 4200
    prominence = 0.05
    height = 0.4
    n_clusters_list = [10,40,70,100,130,160]

elif string == 'MUSIC':
    k_sizes = [10,8,6]
    strides = [3,3,3]
    batch_size = 1024
    epochs1 = 1024
    epochs2 = 8000
    update_interval = 1
    n_clusters = 40
    lr=0.05
    momentum=0.1
    margin = 2400
    window_length = 150
    overlap_percent = 0.50
    window_length2 = 200  
    blackman_window = 40
    prominence = 0.05
    height = 0.1
    print("processing MUSIC")
    time_series = np.loadtxt("./data/Music_Analysis.csv", delimiter=",")
    labels = np.loadtxt("./data/Music_Analysis_labels.csv", delimiter=",")
    n_clusters_list = [10,40,70,100,130,160]

else:
    print("processing SPORTS")
    time_series = np.loadtxt("./data/Sports_Activity.csv", delimiter=",")
    k_sizes = [6,4,4]
    strides = [2,2,2]
    batch_size = 1024
    epochs1 = 1200
    epochs2 = 8000
    update_interval = 1
    n_clusters = 100
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
    n_clusters_list = [10,40,70,100,130,160]

ground_truth = np.where(labels[:-1] != labels[1:])[0]


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

overlap_percent_list = [0.1, 0.2, 0.3, 0.4, 0.5]
coverings_n_clusters = []
f1_n_clusters = []
trained = 0

for overlap_percent in overlap_percent_list:
    f1s = []
    coverings = []
    for loop in range(5):
        print("=================================overlap_percent:",overlap_percent)
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
        batch_size = subsequences.shape[0]
        if trained == 0:
            deep_cluster_model = DEC(n_clusters = n_clusters, input_shape=input_shape,k_sizes=k_sizes,strides=strides)
            deep_cluster_model.to(device)
            pretraining(model=deep_cluster_model, dbgenerator=data_loader, batch_size=batch_size, epochs=epochs1)
            torch.save(deep_cluster_model.state_dict(), './'+string+'_overlap_pretrain.pth')
            trained = 1
        else:
            deep_cluster_model = DEC(n_clusters = n_clusters, input_shape=input_shape,k_sizes=k_sizes,strides=strides)
            deep_cluster_model.load_state_dict(torch.load('./'+string+'_overlap_pretrain.pth'))
            deep_cluster_model.to(device)

        cluster_assigments = refine_clusters(n_clusters, data_loader, deep_cluster_model, device, epochs2, batch_size, lr, momentum, 1)
        cluster_assignments = cluster_assigments

        l = np.unique(cluster_assignments)
        while ( (not all(l[i] == l[i+1] -1 for i in range(len(l) - 1))) or (l[0] != 0) ):
            for assignment in range(np.max(cluster_assignments)+1):
                if assignment not in cluster_assignments:
                    indx = np.where(cluster_assignments > assignment)
                    cluster_assignments[indx] = cluster_assignments[indx]-1
            l = np.unique(cluster_assignments)
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
        #cluster_assignmentss = np.concatenate((cluster_assignments[:window_length2//2], cluster_assignments), axis=None)
        #cluster_assignmentss = np.concatenate((cluster_assignmentss, cluster_assignments[-window_length2//2:]), axis=None)
        similarities = get_label_score(cluster_assignments, window_length2)
        data = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

        data = smooth(np.blackman(blackman_window), data)

        peaks, peak_data = find_peaks(data, height=height,distance=window_length2//3,prominence=prominence)
        real_peaks = peaks + window_length2//2
        predictions = get_changepoints(real_peaks, window_length, overlap_percent)




        print('covering score:',covering(ground_truth, predictions, len(labels)))
        coverings.append(covering(ground_truth, predictions, len(labels)))
        print('f_measure score:',f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=True))
        f1s.append(f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=True)[0])

        print("========================================================")
    coverings = np.asarray(coverings)
    f1s = np.asarray(f1s)
    coverings_n_clusters.append(coverings)
    f1_n_clusters.append(f1s)

coverings_n_clusters = np.asarray(coverings_n_clusters)
f1_n_clusters = np.asarray(f1_n_clusters)

print(coverings_n_clusters)
print(f1_n_clusters)

print()
for i,c in enumerate(coverings_n_clusters):
    print(np.mean(c))
print()    
for i,f in enumerate(f1_n_clusters):
    print(np.mean(f))