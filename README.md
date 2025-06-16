# Deep Clustering for Time Series Segmentation (DC-TSS)

Welcome to the respository for Deep Clustering Time Series Segmentaiton (DC-TSS)! A Time Series Segementaiton method designed for comple large-scale time series datasets. DC-TSS uses a three phase deep learning approach to cluster subsequences of a time series and analzes the clusters for TSS.

## About the Data

### Dataset Sources:

Sports: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

PAMAP2: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

EEG: https://physionet.org/content/eegmmidb/1.0.0/

Music: https://archive.ics.uci.edu/datasets?search=FMA:%20A%20Dataset%20For%20Music%20Analysis

WESAD: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

UV Repository: https://sites.google.com/view/ts-clasp

These datasets are also available in their preprocessed forms:
https://drive.google.com/drive/folders/1ECwHJetl8EPRkQSMD-rLuWimJ5kMp8qW

### Dataset Descriptions

Our dataset is comprised of two real and two semi-real. The real datasets are continuously recorded and have ground truth established from external information. Our semi-real datasets are constructed from segments of real MTS data. These segments are logically concatenated together to mimic continuously recorded data to create a large-scale MTS with several regime changes. For example, a dataset may have multiple 30-second recordings of someone walking and using various gym equipment. We concatenate the 30-second segments of gym equipment usage into 15 minute periods with 2 minute walking periods in between. 

-PAMAP2 is a real MTS dataset with 40 variables, 376,417 observations, and 25 regime changes. PAMAP2 records 12 different physical activities performed by a single person from several sensors sampled at 100 Hertz (Hz).

-Daily and Sports Activity (Sports) is a semi-real MTS dataset with 45 variables, 170,250 observations, and 29 regime changes. Sports is constructed from smaller labeled MTS segments from 17 different activities recorded at 25Hz by various motion capture sensors on a person.

-EEG is a real MTS dataset with 64 variables, 259,520 observations, and 13 regime changes. The dataset contains brain wave recordings of a person performing motor/imagery tasks sampled at 160Hz.

-Music is a semi-real MTS dataset with 148 variables, 309,972 observations, and 29 regime changes. The Music dataset is constructed from 30 different songs from various genres and artists and is the power spectral representation of audio recordings sampled at 44,100Hz

-WESAD is a real physiological and motion MTS dataset. WESAD contain 14 variables, 4496100 to 4949700 observations, and have 16 regime changes. The data is sampled at 700Hz.

-The UV repository was established by Schäfer(https://sites.google.com/view/ts-clasp) and is a collection of 98 univariate TS datasets from various domains. These TS datasets are much shorter than Sports, PAMAP2, EEG, Music with fewer CPs to detect.

### Dataset Preprocessing

All datasets are z-normalized. We use linear interpolation between the last observation and the next obersvation that were recorded. Any variables recorded at lower frequencies have values duplication to match variables recorded at the highest frequency.

## Evaluation Details:
The equation below defines the $F_1$-score where $P$ and $R$ are precision and recall respectively. TP, TN, FP, and FN are true positive, true negative, false positive, and false negative respectively.  A true positive (TP) is determined based on a margin of error, $E$. If the difference between the ground truth and CP is within $E$ then it is labeled as a $TP$ unless a nearer CP is present. We set $E = 225, 2000, 4000,$ and $1800$ for Sports, PAMAP2, EEG, and Music respectively. 

$F_1 = \frac{2 P R}{P + R}, P = \frac{TP}{TP+FP}, R = \frac{TP}{TP + FN}$

BOCPD and ClaSP are not explicitly designed to handle MTS datasets. In order to evaluate them on MTS, we run the methods on each variable and refine their results. Our refinement runs an instance of the algorithm on each variable. The CPs returned by each instance are combined into one final result by removing duplicates and averaging CPs in close proximity. We determine if CPs are in close proximity if their difference is less than the margin of error $E$. For example, CP results from two variables may be $[25, 68]$ and $[25, 66, 120]$, combined they form $[25, 67, 120]$ given $E=2$. The intuition here is that a change in one variable is a change in the whole system.


## To Run:
The jupyternotebook displays a step by step tutorial for segmenting the PAMAP2 TS dataset using the provided .py files.

Evalution Metric Details:

