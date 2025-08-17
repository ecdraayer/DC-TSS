import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings
import time, copy

import utils
import TIRE
import simulate

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("gt")
    parser.add_argument("WinSize")
    parser.add_argument("outputfile")
    
    args = parser.parse_args()

    ts_file = args.data
    gt = args.gt
    window_size = args.WinSize
    outfile = args.outputfile
    # For PAMAP2 set margin = 30*100
    # For WESAD set margin = 700*30
    
    
    #ts_file = "\\Users\\Erik\\Documents\\Clustering_MTS_Research\\_MTS_Clustering\\data\\wesad_data_9.csv"
    labels = np.loadtxt(gt, delimiter=",")
    ground_truth = np.where(labels[:-1] != labels[1:])[0]
    time_series = np.loadtxt(ts_file, delimiter=",")
    
    domain = "both" #choose from: TD (time domain), FD (frequency domain) or both

    #parameters TD
    intermediate_dim_TD=0
    latent_dim_TD=1 #h^TD in paper
    nr_shared_TD=1 #s^TD in paper
    K_TD = 2 #as in paper
    nr_ae_TD= K_TD+1 #number of parallel AEs = K+1
    loss_weight_TD=1 #lambda_TD in paper

    #parameters FD
    intermediate_dim_FD=10
    latent_dim_FD=1 #h^FD in paper
    nr_shared_FD=1 #s^FD in paper
    K_FD = 2 #as in paper
    nr_ae_FD=K_FD+1 #number of parallel AEs = K+1
    loss_weight_FD=1 #lambda^FD in paper
    nfft = 30 #number of points for DFT
    norm_mode = "timeseries" #for calculation of DFT, should the timeseries have mean zero or each window?
    
    windows = utils.ts_to_windows(timeseries, 0, window_size, stride=1)
    windows = utils.minmaxscale(windows,-1,1)
    #windows = utils.combine_ts(windows)
    windows_TD = windows

    t1_start = process_time() 
    change_points = []
    disses = []
    for i in range(len(data)):
        timeseries = data[i]
        windows = utils.ts_to_windows(timeseries, 0, window_size, stride=1)
        windows = utils.minmaxscale(windows,-1,1)
        windows_TD = windows
        windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
        shared_features_TD = TIRE.train_AE(windows_TD, intermediate_dim_TD, latent_dim_TD, nr_shared_TD, nr_ae_TD, loss_weight_TD)
        shared_features_FD = TIRE.train_AE(windows_FD, intermediate_dim_FD, latent_dim_FD, nr_shared_FD, nr_ae_FD, loss_weight_FD)
        dissimilarities = TIRE.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, domain, window_size)
        disses.append(dissimilarities)
        change_point_scores = TIRE.change_point_score(dissimilarities, window_size)
        change_points.append(change_point_scores)
    t1_stop = process_time()

    change_points = np.array(change_points)
    disses = np.array(disses)
    
    np.savetxt(outfile+"_TIRE_change_points.csv", change_points, delimiter=",")
    np.savetxt(outfile+"_TIRE_disses..csv", disses, delimiter=",")


if __name__ == "__main__":
    main()