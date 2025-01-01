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
    print(multiprocessing.cpu_count())
    print(string)
    if string  == 'PAMAP2':
        print("processing PAMAP2")
        time_series = np.loadtxt("./data/PAMAP2.csv", delimiter=",")
        hazard = 3600
    elif string == 'EEG':
        print("processing EEG")
        time_series = np.loadtxt("./data/EEG_short.csv", delimiter=",")
        hazard = 20000
    elif string == 'MUSIC':
        print("processing MUSIC")
        time_series = np.loadtxt("./data/Music_Analysis.csv", delimiter=",")
        hazard = 20000
    else:
        print("processing SPORTS")
        time_series = np.loadtxt("./data/Sports_Activity.csv", delimiter=",")
        hazard = 2500

    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(hazard), bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))
    selected_features = np.arange(int(start), int(stop)+1, 1)

    for i,ts in enumerate(time_series):
        # Online estimation and get the maximum likelihood r_t at each time point
        if i not in selected_features:
            continue        
        t1_start = time.time()
        rt_mle = np.empty(ts.shape)
        for j, d in enumerate(ts):
            bc.update(d)
            rt_mle[j] = bc.rt 
        t1_stop = time.time()
        print('time to process feature ' + str(i) + ':', t1_stop - t1_start) 
        print(i, ts.shape)
        
        filename = "/fs1/home/edraayer/Bocpd_Results/" + string + "/" + str(i) + "_iteration.out"
        np.savetxt(filename, rt_mle, delimiter=',')
           

if __name__ == "__main__":
    main()