# User Manual


## Introduction

The following manual is a user-friendly documentation that describes all the resources and dependencies that an user have to meet in order to succesfully use the library.

## Signal data assumptions

this module does not cover the signals adquisition, it is designed to process signal data previously acquired, by contrast, the correct operation of the software requires key assumptions about the nature of this data:

- The raw signals must be 8 channels superficial electromyography signals sampled at 200 Hz. This development used the Myo Armband as adquisition sensor.
-The signal file is saved as an .mat extension file. It is due the Myo armband interface used for this particular implementation needed Mathworks Matlab. 
- The signals adquisition process is governed by the protocol presented in the annexes directory. It has important implications over the signal processing since the segmentation process is time-based and expects the execution times of the protocol.
- The signals adquisition protocol divides the data collection in 3 sessions with different levels of muscular fatigue where level 1 is the lighter and level 3 the stronger.
- The gestures that are supported by this specific implementation are:

    - **thumb-index pincer (TIP):**
        Movement where the thumb and the index finger touch the tips of each of them.

    - **thumb-Middle pincer (TMP):**
        Movement where the thumb and the middle finger touch the tips of each of them.

    - **thumb-ring pincer (TRP):**
        Movement where the thumb and the ring finger touch the tips of each of them.

    - **thumb-pinky pincer (TPP):**
        Movement where the thumb and the pinky finger touch the tips of each of them.

    - **closed hand (CH):**
        Movement where the whole hand closes itself into a fist.

    - **rest hand (RH):**
        hand state where the hand is not in any movement.

- The module is design to process the signals saved within a directory with a specific structure:

    - **Root:**  The root directory, whatever its name may be, represents the main directory that contains all the user directories. 

    - **userx:** The user directories contains all the signals from an specific subject where x is an integer that identifies the subject uniquely. The system expects the directory name to be user always, the only difference between the users directory names are the values of x.

    - **gesture** The gestures directory pretend to save the 3 levels of fatigue of a specific gesture. Each user directory should contain 5 gestures directories whose names corresponds to the list of abbreviations presented previously: {TIP, TMP, TRP ,TPP, CH }. Note that the RH gesture is missed in this last list, this is because this gesture has not a session assigned, rather, it is extracted from the resting periods of all the other sessions. 

    - **fatigue:** The fatigue directory contains the superficial electromyography signals of a specific level of fatigue. Each gesture directory should save 3 levels of fatgue whose names correspond to a numbre between 1 and 3. 

    - **emg_signal.mat:** This is the signal of the subject executing a particular gesture with an specific degree  of fatigue.

```
  /   
    -userx/
        -gesture/
            -fatigue/
                -emg_signal.mat
          
```


## First steps

This module is conceived as a software library to be used for developers in movement intention identification applications. Bellow it is described the step by step process to use this software for the first time. 

It is assumed that the user has all the tools and configurations for python development in his own local computer and the acquired signals must meet all the key asumptions previously described. Bellow are listed the library dependencies of this module:

```
- numpy   2.0.0
- pandas   2.2.2
- matplotlib   3.9.1
- scikit-learn   1.5.1
- plotly     5.23.0
- scipy     1.14.0 
- seaborn    0.13.2
- mrmr_selection    0.2.8  
```

1. Download the library from the repository: `https://github.com/portafolioGomezan/gestureRecognition` or use the command `git clone https://github.com/portafolioGomezan/gestureRecognition.git` to clone the module to your local computer.

2. Once the developer have access to the code, already is able to import the modules and consume its resources. It is recommended to read the documentation of each of the entities that are part of the module. The description documentation explains in detail all the methods between the classes. 

3. To observe some forms of implementations, the example bellow guides the user through a very basic beginner example; aditionally, the code repository is equipped with some examples for each class and a video tutorial that illustrates the whole process of downloading and setting up of the software. 

## Example

The next code is a simple example illustrates how to plot the sample of a raw signal taken from a subject. 

```python
import signalProcessor as sp
import pandas as pd
import scipy.io

def main():
  """
  Main function for the example usage of the SignalProcessor class.
  """
  # Se crea un dataset con la función 'crearDataset()
  signalProcessor = sp.SignalProcessor()

  #Se grafica una señal para verificar que el modulo funciona correctamente
  mat= scipy.io.loadmat(r"ruta/signalTest/user1/TIP/2/emg_TIP.mat")
  print(type(mat))  
  signal=mat['emgSignalOut']
  print(type(signal))
  emg = pd.DataFrame(signal)
  print(len(emg))
  emg.head()

  signalProcessor.graficar(emg)
```
