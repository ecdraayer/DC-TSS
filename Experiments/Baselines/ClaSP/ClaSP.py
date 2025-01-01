import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
import csv
from scipy import stats as st
from utils import *

import numpy as np
import matplotlib.pyplot as plt
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
from time import process_time
from claspy.segmentation import BinaryClaSPSegmentation
import sys
import argparse
import multiprocessing
import bocd

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("start")
    parser.add_argument("stop")
    args = parser.parse_args()

    string = args.data
    start = args.start
    stop = args.stop

    if string  == 'PAMAP2':
        print("processing PAMAP2")
        time_series = np.loadtxt("./data/PAMAP2.csv", delimiter=",")
    elif string == 'EEG':
        print("processing EEG")
        time_series = np.loadtxt("./data/EEG_short.csv", delimiter=",")
    elif string == 'MUSIC':
        print("processing MUSIC")
        time_series = np.loadtxt("./data/Music_Analysis.csv", delimiter=",")
        noise = np.random.normal(0,0.1,time_series.shape)
        time_series = time_series + noise
    else:
        print("processing SPORTS")
        time_series = np.loadtxt("./data/Sports_Activity.csv", delimiter=",")

    clasp = BinaryClaSPSegmentation()
    selected_features = np.arange(int(start), int(stop)+1, 1)
    

    predictions = []
    for i,ts in enumerate(time_series):
        if i not in selected_features:
            continue   
        print(i, ts.shape)
        t1_start = process_time() 
        predict = clasp.fit_predict(ts)
        predictions.append(predict)
        t1_stop = process_time()
        filename = string+'_Partitions/' + str(i) + "_iteration.out"
        np.savetxt(filename, predict, delimiter=',')
        print('time to process feature ' + str(i) + ':', t1_stop - t1_start)
        print('predictions for feature ' + str(i) + ':', predict)
    #predictions = np.hstack(predictions)

    #print('\nall predictions: \n')
    #filename = string+'_Partitions/'+'all_iterations.out'
    #np.savetxt(filename, predictions, delimiter=',')
    

if __name__ == "__main__":
    main()