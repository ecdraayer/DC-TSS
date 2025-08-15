import numpy as np
import ruptures as rpt
from time import process_time
import sys
import argparse
from utils import *
import os

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("gt")
    parser.add_argument("margin")
    
    args = parser.parse_args()

    ts_file = args.data
    gt = args.gt
    margin = args.margin
    # For PAMAP2 set margin = 30*100
    # For WESAD set margin = 700*30
    
    
    #ts_file = "\\Users\\Erik\\Documents\\Clustering_MTS_Research\\_MTS_Clustering\\data\\wesad_data_9.csv"
    labels = np.loadtxt(gt, delimiter=",")
    ground_truth = np.where(labels[:-1] != labels[1:])[0]
    time_series = np.loadtxt(ts_file, delimiter=",")
    time_series = time_series.T
    
    assert time_series.shape[1] < time_series.shape[0], "Need to transpose time series"
    
    t1_start = process_time() 
    algo = rpt.Pelt(model=model, min_size=2000, jump=10).fit(time_series)  
    predictions = algo.predict(pen=np.log(time_series.shape[0]) * time_series.shape[1] * 10)
    predictions = predictions[:-1]
    t1_stop = process_time()

    print('time to process TS: ', t1_stop - t1_start)
    print('predictions for feature: ', predictions)
    print(len(predictions))
    print('margin:',margin)
    print('covering score:',covering(ground_truth, predictions, time_series.shape[0]))
    print('f_measure score:',f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=True))

if __name__ == "__main__":
    main()