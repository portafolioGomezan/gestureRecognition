# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.fft

"""codigo relacionado con la grafica de señales"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler

from mrmr import mrmr_classif
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import welch


class FeatureSelector:
  """
    A class that represents a feature selector.

    Attributes:
    GestosInt (dict): A dictionary mapping gesture names to their corresponding integer values.
    IntGestos (dict): A dictionary mapping integer values to their corresponding gesture names.
    nombresCarac (list): A list of feature names.
    nombresFull (list): A list of feature names with index numbers.
    nombresFullIndex (range): A range of index numbers for feature names.
    IntGestosFull (dict): A dictionary mapping index numbers to feature names.

    Methods:
    __init__(): Initializes the FeatureSelector object.
    calcular_fft(señal): Calculates the Fast Fourier Transform (FFT) of a signal.
    graficarPCA(strP): Generates a 3D scatter plot using PCA data.
    clustering(dataSet): Clusters the given dataset based on a specified feature.
    plot_matrix(data, titulo): Plots a matrix of features with a given title.
    getRMS(senal): Calculates the root mean square (RMS) of a signal.
    getSTD(senal): Calculates the standard deviation of a signal.
    getVarianza(senal): Calculates the variance of a signal.
    getMAV(senal): Calculates the Mean Absolute Value (MAV) of a signal.
    getWL(senal): Calculates the waveform length (WL) of a signal.
    getPromedio(senal): Calculates the average of a signal.
    zero_crossing_count(signal): Counts the number of zero crossings in a signal.
    calculate_kurtosis(signal): Calculates the kurtosis of a signal.
    calculate_skewness(signal): Calculates the skewness of a signal.
    getIemg(signal): Calculates the Integrated EMG (IEMG) of a signal.
    getSSC(signal): Calculates the number of sign changes in a signal.
    getWAMP(signal): Calculates the Willison amplitude in a signal.
    getMAS(signal_data): Calculates the frequency with the highest amplitude (MAS) using the Fast Fourier Transform (FFT).
    getMP(signal): Calculates the average power spectrum using the Welch method.
  """

  class FeatureSelector:
    def __init__(self):
      """
      Initializes the FeatureSelector class.

      The FeatureSelector class is used to manage gesture features and their corresponding indices.

      Attributes:
      - GestosInt: A dictionary mapping gesture names to their corresponding indices.
      - IntGestos: A dictionary mapping gesture indices to their corresponding names.
      - nombresCarac: A list of gesture names.
      - nombresFull: A list of gesture names with index numbers appended.
      - nombresFullIndex: A range of index numbers for the gesture names.
      - IntGestosFull: A dictionary mapping index numbers to gesture names with index numbers appended.
      """
      self.GestosInt = {
        "RMS": 0, "STD": 1, "Varianza": 2, "MAV": 3, "WL": 4, "Promedio": 5, "ZC": 6, "kurtosis": 7, "skewness": 8,
        "iEMG": 9, "SSC": 10, "WAMP": 11, "MAS": 12, "MP": 13, "MDF": 14, "MNF": 15
      }

      self.IntGestos = {
        0: "RMS", 1: "STD", 2: "Varianza", 3: "MAV", 4: "WL", 5: "Promedio", 6: "ZC", 7: "kurtosis", 8: "skewness",
        9: "iEMG", 10: "SSC", 11: "WAMP", 12: "MAS", 13: "MP", 14: "MDF", 15: "MNF"
      }

      self.nombresCarac = ["RMS", "STD", "Varianza", "MAV", "WL", "Promedio", "ZC", "kurtosis", "skewness",
                 "iEMG", "SSC", "WAMP", "MAS", "MP", "MDF", "MNF"]

      self.nombresFull = [j + str(i) for i in range(8) for j in self.nombresCarac]
      self.nombresFullIndex = range(8 * len(self.nombresCarac))

      self.IntGestosFull = dict(zip(self.nombresFullIndex, self.nombresFull))

  def calcular_fft(señal):
    """
    Calcula la Transformada Rápida de Fourier (FFT) de una señal.

    Parameters:
    señal (array-like): La señal de entrada.

    Returns:
    array-like: El resultado de la FFT aplicada a la señal.
    """
    # Aplicar la FFT a la señal
    fft_resultado = scipy.fft.fft(señal)

    return fft_resultado

  def graficarPCA(strP):
    """
    Generates a 3D scatter plot using PCA data.

    Parameters:
    strP (list): A list of numpy arrays containing the PCA data for each feature.

    Returns:
    None
    """

    simbolos=['circle', 'circle-open', 'cross', 'diamond','diamond-open', 'square', 'square-open', 'x']
    color=['rgba(0, 255, 255, 0.5)','rgba(255, 0, 0, 0.5)','rgba(0, 0,255, 0.5)','rgba(0, 255, 0, 0.5)','rgba(255, 255, 0, 0.5)']
    nombres=["CH","TIP","TMP","TRP","TPP"]

    # Creating the 3D scatter plot
    fig = go.Figure()

    for i in range(1,6):
      data=strP[i]
      xdata=data[:,0]
      ydata=data[:,1]
      zdata=data[:,2]
      fig.add_trace(go.Scatter3d(x=xdata, y=ydata, z=zdata, mode='markers', marker=dict(size=6, color=color[i-1], symbol=simbolos[i-1]), name=nombres[i-1]))

    # Customizing the layout
    fig.update_layout(title='3D Scatter Plot Example',
                    scene=dict(xaxis_title='x axis', yaxis_title='y axis', zaxis_title='z axis'),
                    showlegend=True)

    fig.show()

  def clustering(dataSet):
    """
    Clusters the given dataset based on a specified feature.

    Parameters:
    dataSet (DataFrame): The dataset to be clustered.

    Returns:
    dict: A dictionary containing the clustered datasets, where the keys are the distinct values of the specified feature.

    """
    valy = "0"

    estructuras = {}
    for caracteristica, grupo in dataSet.groupby(valy):
      estructuras[caracteristica] = grupo.drop(valy, axis=1)

    return estructuras

  def plot_matrix(self, data, titulo):
    """
    Plots a matrix of feature with a given title.

    Args:
      data (list): A list of 128 data points.
      titulo (str): The title of the plot.

    Raises:
      ValueError: If the length of the data is not equal to 128.

    Returns:
      None
    """
    if len(data) != 128:
      raise ValueError("Es necesario recibir 128 caracteristicas")

    matrix = np.array(data).reshape(8, 16)

    plt.figure(figsize=(20, 10))

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Values')
    plt.title(titulo)
    plt.xlabel('Caracteristicas')
    plt.ylabel('Canales')

    plt.yticks(range(1,9))
    plt.xticks(np.arange(16), self.nombresCarac)

    plt.show()


  #Mediana de la frecuencia (MDF)
  def getMDF(signal):
    """
    Calculate the Median Frequency (MDF) of a given signal.

    Parameters:
    signal (array-like): The input signal.

    Returns:
    float: The median frequency of the signal.
    """

    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal)
    median_frequency = np.median(magnitude)

    return median_frequency

  #Promedio de la frecuencia (MNF)
  def getMNF(signal):
    """
    Calculate the mean normalized frequency of a given signal.

    Parameters:
    signal (array-like): The input signal.

    Returns:
    float: The mean normalized frequency of the signal.
    """

    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal)
    mean_frequency = np.mean(magnitude)

    return mean_frequency



  #Descomposición completa
  def descomposicion(self,senal):
    """
    This function takes a signal as input and applies a list of functions to decompose the signal into its features.

    Parameters:
    - senal: The input signal to be decomposed.

    Returns:
    - A list of decomposed features of the input signal.
    """
    listaCaracteristicas = [self.getRMS, self.getSTD,self.getVarianza,self.getMAV,self.getWL, self.getPromedio,
                          self.zero_crossing_count, self.calculate_kurtosis, self.calculate_skewness, self.getIemg, self.getSSC, self.getWAMP,
                          self.getMAS, self.getMP, self.getMDF, self.getMNF]

    return [funcion(senal) for funcion in listaCaracteristicas]

  def caracterizar(self,df):
    """
    Caracterizes the given DataFrame by extracting windows from each signal.

    Parameters:
    df (DataFrame): The input DataFrame containing signals.

    Returns:
    DataFrame: A new DataFrame containing the extracted windows.
    """

    # Crear una lista vacía
    ventanas=[]

    # Iterar a través de las señales y extraer las ventanas
    for i in range(len(df)):
      fila = list(df.loc[i])
      ventanas.append(self.descomposicion(fila))

    return pd.DataFrame(ventanas)


