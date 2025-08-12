import numpy as np

import time
import argparse
import bocd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("output")
    parser.add_argument("start")
    parser.add_argument("stop")
    parser.add_argument("alpha")
    parser.add_argument("beta")
    parser.add_argument("kappa")
    parser.add_argument("hazard")
    args = parser.parse_args()

    ts_file = args.data
    output_name = args.output
    feature_index_1 = args.start
    feature_index_2 = args.stop
    alpha = args.alpha
    beta = args.beta
    kappa = args.kappa
    hazard = args.hazard
    
    time_series = np.loadtxt(ts_file, delimiter=",")
    

    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(hazard), bocd.StudentT(mu=0, kappa=kappa, alpha=alpha, beta=beta))
    selected_features = np.arange(int(feature_index_1), int(feature_index_2)+1, 1)

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
        
        filename = "/"+output_name+"/Bocpd_Results_feature_" + str(i) + "_iteration.out"
        np.savetxt(filename, rt_mle, delimiter=',')
           

if __name__ == "__main__":
    main()