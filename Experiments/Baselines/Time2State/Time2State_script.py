from Time2State.time2state import Time2State
from Time2State.adapers import CausalConv_LSE_Adaper
from Time2State.clustering import DPGMM
from Time2State.default_params import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from TSpy.view import plot_mts
import matplotlib.pyplot as plt
import numpy as np
import os

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("gt")
    parser.add_argument("margin")
    parser.add_argument("output_file")
    
    args = parser.parse_args()

    ts_file = args.data
    gt = args.gt
    margin = args.margin
    out_file = args.output_file
    # For PAMAP2 set margin = 30*100
    # For WESAD set margin = 700*30
    
    
    #ts_file = "\\Users\\Erik\\Documents\\Clustering_MTS_Research\\_MTS_Clustering\\data\\wesad_data_9.csv"
    labels = np.loadtxt(gt, delimiter=",")
    ground_truth = np.where(labels[:-1] != labels[1:])[0]
    time_series = np.loadtxt(ts_file, delimiter=",")
    time_series = time_series.T
    
    # model
    params_LSE['in_channels'] = data.shape[1]
    params_LSE['out_channels'] = 14
    params_LSE['nb_steps'] = 5
    params_LSE['win_size'] = win_size
    params_LSE['win_type'] = 'hanning' # {rect, hanning}
    
    step = data.shape[0]//50
    start = time.time()
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None), params_LSE)
    t2s.fit(data, win_size, step)
    end = time.time()
    print(end-start)
    predictions =  np.where(t2s.state_seq[:-1] != t2s.state_seq[1:])[0]
    print(predictions)
    numpy.savetxt(out_file, predictions, delimiter=",")

if __name__ == "__main__":
    main()