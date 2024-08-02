# Index


## Welcome to iREHAB movement identification module

The iREHAB project is an initiative dedicated to the intelligent and progressive rehabilitation of hand motor function in post-ACV patients. Currently, the iREHAB project is developing an active assistive device. The processing of the sEMG signals, as well as the execution of the gesture recognition algorithm, must be carried out within an online embedded system. Once a sample is taken, it should be immediately processed, thus avoiding the need to store or transmit samples to a secondary device for processing.

This module is designed to support all the processes witin the feature engineering pipeline for a gesture recognition application. It includes the signal processing, feature extraction and feature selection.


## Project layout

    signalProcessor.py    # class responsible for the processing of raw EMG signals in segmented windows according to the 6 hand movement reference gestures.
    featureSelector.py    # class in charge of calculating a total of 16 features available from the processing windows available by the SignalProcessor.  .
    featureExplorer.py    # class capable of calculating a wide variety of tests on the characteristics of a dataset, including tests for feature importance, variability, PCA and etc.
    
