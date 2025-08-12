import numpy as np
import ruptures as rpt
from time import process_time
import sys
import argparse
import stumpy
from stumpy.floss import _cac

# See https://stumpy.readthedocs.io/en/latest/Tutorial_Semantic_Segmentation.html for more details

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("feature_index_1")
    parser.add_argument("feature_index_2")
    parser.add_argument("output")
    parser.add_argument("m")

    args = parser.parse_args()

    ts_file = args.data
    output_name = args.output
    m = args.m
    feature_index_1 = args.feature_index_1
    feature_index_2 = args.feature_index_2
    time_series = np.loadtxt(ts_file, delimiter=",")
    selected_features = np.arange(int(feature_index_1), int(feature_index_2)+1, 1)

    # Online estimation and get the maximum likelihood r_t at each time point      
    for i,ts in enumerate(time_series):
        # Online estimation and get the maximum likelihood r_t at each time point
        if i not in selected_features:
            continue        
    t1_start = process_time() 
    mp = stumpy.stump(time_series[i], m=m)
    L = m
    cac, regime_locations = stumpy.fluss(mp[:, 1], L=L, n_regimes=2, excl_factor=1)
    t1_stop = process_time()
    filename = "/"+output_name+"/FLOSS_CAC_Scores_featuer"+str(i)+".out"
    np.savetxt(filename, cac, delimiter=',')
    print('time to process feature: ', t1_stop - t1_start)
    print('predictions for feature: ', cac)
  
    

if __name__ == "__main__":
    main()