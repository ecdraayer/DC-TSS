import numpy as np
from time import process_time
from claspy.segmentation import BinaryClaSPSegmentation
import sys
import argparse


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("output")
    parser.add_argument("start")
    parser.add_argument("stop")
    args = parser.parse_args()

    ts_file = args.data
    output_name = args.output
    feature_index_1 = args.start
    feature_index_2 = args.stop
    
    time_series = np.loadtxt(ts_file, delimiter=",")

    clasp = BinaryClaSPSegmentation()
    selected_features = np.arange(int(feature_index_1), int(feature_index_2)+1, 1)
    

    predictions = []
    for i,ts in enumerate(time_series):
        if i not in selected_features:
            continue   
        print("Processing feature: ", i)
        t1_start = process_time() 
        predict = clasp.fit_predict(ts)
        predictions.append(predict)
        t1_stop = process_time()
        filename = output_name+'_ClaSP_Partitions/' + str(i) + "_iteration.out"
        np.savetxt(filename, predict, delimiter=',')
        #times.append(t1_stop - t1_start)
        print('time to process feature ' + str(i) + ':', t1_stop - t1_start)
        print('predictions for feature ' + str(i) + ':', predict)

    

if __name__ == "__main__":
    main()