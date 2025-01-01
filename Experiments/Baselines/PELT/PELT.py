import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
import csv
from scipy import stats as st
from utils import *

import numpy as np
import os
import math
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance


from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_distance
from typing import Optional, List
from scipy.io import arff

from scipy.signal import find_peaks
import ruptures as rpt
import time
import sys
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    string = args.data
    print(string)
    if string  == 'PAMAP2':
        print("processing PAMAP2")
        time_series = np.loadtxt("./data/PAMAP2.csv", delimiter=",")
        labels = np.loadtxt("./data/PAMAP2_labels.csv", delimiter=",")
        pens = [24000]
        jump = 25
        margin = 3000
    elif string == 'EEG':
        print("processing EEG")
        time_series = np.loadtxt("./data/EEG.csv", delimiter=",")
        labels = np.loadtxt("./data/EEG_labels.csv", delimiter=",")
        pens = [6800, 7100, 7500]
        margin = 4200

    elif string == 'MUSIC':
        print("processing MUSIC")
        time_series = np.loadtxt("./data/Music_Analysis.csv", delimiter=",")
        labels = np.loadtxt("./data/Music_Analysis_labels.csv", delimiter=",")
        jump = 25
        pens = [80000]
        margin = 1800

    else:
        print("processing SPORTS")
        time_series = np.loadtxt("./data/Sports_Activity.csv", delimiter=",")
        labels = np.loadtxt("./data/Sports_Activity_labels.csv", delimiter=",")
        pens = [6500, 7000]
        margin = 200

    ground_truth = np.where(labels[:-1] != labels[1:])[0]


    
    for pen in pens:
        t1_start = time.time()
        model = "l2"  # "l2", "rbf"
        algo = rpt.Pelt(model=model, min_size=2000, jump=jump).fit(time_series.T)  
        predictions = algo.predict(pen=pen)
        t1_stop = time.time()
        print("--- %s seconds ---" % (t1_stop-t1_start))
        predictions = predictions[:-1]
        print(predictions)
        print('penalty:', pen)
        print('covering score:',covering(ground_truth, predictions, len(labels)))
        print('f_measure score:',f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=True))
        
        filename = string+"_pelt_predictions_"+str(pen)+".csv"
        np.savetxt(filename, predictions, delimiter=',')
     

if __name__ == "__main__":
    main()