import numpy as np
import ruptures as rpt
from time import process_time
import sys
import argparse


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("output")

    args = parser.parse_args()

    ts_file = args.data
    output_name = args.output

    time_series = np.loadtxt(ts_file, delimiter=",")

    # Online estimation and get the maximum likelihood r_t at each time point       
    t1_start = process_time() 
    algo = rpt.Window(model="l2").fit(time_series)
    predictions = algo.predict(pen=np.log(time_series.shape[0]) * time_series.shape[-1] * 1**2)
    t1_stop = process_time()
    filename = "/"+output_name+"/Window_Results.out"
    np.savetxt(filename, predictions, delimiter=',')
    print('time to process feature: ', t1_stop - t1_start)
    print('predictions for feature: ', predictions)
  
    

if __name__ == "__main__":
    main()