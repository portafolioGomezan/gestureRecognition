# -*- coding: utf-8 -*-

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SignalProcessor:
  """
  Class for processing EMG signals.

  Attributes:
  - tn1 (int): Time for level 1 fatigue (in seconds)
  - tn2 (int): Time for level 2 fatigue (in seconds)
  - tn3 (int): Time for level 3 fatigue (in seconds)
  - inicio (int): Duration of the start zone (in seconds)
  - gesto (int): Duration of the gesture zone (in seconds)
  - descp (int): Duration of the short rest zone after each gesture (in seconds)
  - descm (int): Duration of the medium rest zone after each gesture (in seconds)
  - descl (int): Duration of the long rest zone after each gesture (in seconds)
  - fatiga (int): Duration of the fatigue zone after each gesture (in seconds)
  - foco (int): Duration of the focus zone after each gesture (in seconds)
  - gestos (list): List of supported gestures
  - gestos_int (dict): Dictionary mapping supported gestures to integers
  - nivel1g (list): List of time intervals for level 1 gestures
  - nivel2g (list): List of time intervals for level 2 gestures
  - nivel3g (list): List of time intervals for level 3 gestures
  - niveles (list): List of supported levels
  - formato_nivel (dict): Dictionary mapping levels to their respective gesture time intervals
  - formato_descanso (dict): Dictionary mapping levels to their respective rest time intervals
  - tam (int): Window size in milliseconds
  - fs (int): Sampling frequency in Hz
  - over (float): Overlap between Windows

  Methods:
  - __init__: Initializes the object with default values for various parameters.
  - graficar: Plot the EMG signal.
  - separacion: Transpose the dataframe.
  - simplificarGesto: Simplifies the given signal by performing a separation operation.
  - segmentar: Segments a signal into overlapping windows.
  - procesarVentanas: Process the segmented EMG signal windows.
  - buscarGestos: Find gestures in the EMG signal.
  - buscarDescanso: Find rest periods in the EMG signal.
  """

  def __init__(self):
    """
    Initializes the object with default values for various parameters.

    Parameters:
    - tn1 (int): Time for level 1 fatigue (in seconds)
    - tn2 (int): Time for level 2 fatigue (in seconds)
    - tn3 (int): Time for level 3 fatigue (in seconds)
    - inicio (int): Duration of the start zone (in seconds)
    - gesto (int): Duration of the gesture zone (in seconds)
    - descp (int): Duration of the short rest zone after each gesture (in seconds)
    - descm (int): Duration of the medium rest zone after each gesture (in seconds)
    - descl (int): Duration of the long rest zone after each gesture (in seconds)
    - fatiga (int): Duration of the fatigue zone after each gesture (in seconds)
    - foco (int): Duration of the focus zone after each gesture (in seconds)
    - gestos (list): List of supported gestures
    - gestos_int (dict): Dictionary mapping supported gestures to integers
    - nivel1g (list): List of time intervals for level 1 gestures
    - nivel2g (list): List of time intervals for level 2 gestures
    - nivel3g (list): List of time intervals for level 3 gestures
    - niveles (list): List of supported levels
    - formato_nivel (dict): Dictionary mapping levels to their respective gesture time intervals
    - formato_descanso (dict): Dictionary mapping levels to their respective rest time intervals
    - tam (int): Window size in milliseconds
    - fs (int): Sampling frequency in Hz
    - over (float): Overlap between Windows

    Returns:
    None
    """
    # Tiempos de los 3 niveles de fatiga importantes
    self.tn1 = 130
    self.tn2 = 230
    self.tn3 = 190

    # Duración de zonas de muestras (s)
    self.inicio = 3
    self.gesto = 6
    self.descp = 3
    self.descm = 40
    self.descl = 60
    self.fatiga = 50
    self.foco = 11

    # Lista de gestos soportados
    self.gestos = ["Desc", "CH", "TIP", "TMP", "TRP", "TPP"]
    # Relaciona los gestos soportados con números enteros
    self.gestos_int = {"Desc": 0, "CH": 1, "TIP": 2, "TMP": 3, "TRP": 4, "TPP": 5}

    self.nivel1g=[[self.inicio,self.inicio+self.gesto],
     [self.inicio+self.gesto+self.descp,self.inicio+2*self.gesto+self.descp],
     [self.inicio+2*(self.gesto+self.descp),self.inicio+3*self.gesto+2*self.descp],
     [self.inicio+3*(self.gesto+self.descp),self.inicio+4*self.gesto+3*self.descp],
     [self.inicio+4*(self.gesto+self.descp),self.inicio+5*self.gesto+4*self.descp],
     [2*self.inicio+5*self.gesto+4*self.descp+self.descm,2*self.inicio+6*self.gesto+4*self.descp+self.descm],
     [2*self.inicio+6*self.gesto+5*self.descp+self.descm,2*self.inicio+7*self.gesto+5*self.descp+self.descm],
     [2*self.inicio+7*self.gesto+6*self.descp+self.descm,2*self.inicio+8*self.gesto+6*self.descp+self.descm],
     [2*self.inicio+8*self.gesto+7*self.descp+self.descm,2*self.inicio+9*self.gesto+7*self.descp+self.descm],
     [2*self.inicio+9*self.gesto+8*self.descp+self.descm,2*self.inicio+10*self.gesto+8*self.descp+self.descm]]

    self.nivel2g=[[self.inicio+self.fatiga,self.fatiga+self.inicio+self.gesto],
     [self.inicio+self.gesto+self.descp+self.fatiga,self.fatiga+self.inicio+2*self.gesto+self.descp],
     [self.inicio+2*(self.gesto+self.descp)+self.fatiga,self.fatiga+self.inicio+3*self.gesto+2*self.descp],
     [self.inicio+3*(self.gesto+self.descp)+self.fatiga,self.fatiga+self.inicio+4*self.gesto+3*self.descp],
     [self.inicio+4*(self.gesto+self.descp)+self.fatiga,self.fatiga+self.inicio+5*self.gesto+4*self.descp],
     [2*self.inicio+5*self.gesto+4*self.descp+self.descm+2*self.fatiga,2*self.fatiga+2*self.inicio+6*self.gesto+4*self.descp+self.descm],
     [2*self.inicio+6*self.gesto+5*self.descp+self.descm+2*self.fatiga,2*self.fatiga+2*self.inicio+7*self.gesto+5*self.descp+self.descm],
     [2*self.inicio+7*self.gesto+6*self.descp+self.descm+2*self.fatiga,2*self.fatiga+2*self.inicio+8*self.gesto+6*self.descp+self.descm],
     [2*self.inicio+8*self.gesto+7*self.descp+self.descm+2*self.fatiga,2*self.fatiga+2*self.inicio+9*self.gesto+7*self.descp+self.descm],
     [2*self.inicio+9*self.gesto+8*self.descp+self.descm+2*self.fatiga,2*self.fatiga+2*self.inicio+10*self.gesto+8*self.descp+self.descm]]

    self.nivel3g=[[self.inicio+self.fatiga,self.fatiga+self.inicio+self.gesto],
     [self.inicio+self.gesto+self.descp+self.fatiga,self.fatiga+self.inicio+2*self.gesto+self.descp],
     [self.inicio+2*(self.gesto+self.descp)+self.fatiga,self.fatiga+self.inicio+3*self.gesto+2*self.descp],
     [self.inicio+3*(self.gesto+self.descp)+self.fatiga,self.fatiga+self.inicio+4*self.gesto+3*self.descp],
     [self.inicio+4*(self.gesto+self.descp)+self.fatiga,self.fatiga+self.inicio+5*self.gesto+4*self.descp],
     [2*self.inicio+5*self.gesto+4*self.descp+2*self.fatiga,2*self.fatiga+2*self.inicio+6*self.gesto+4*self.descp],
     [2*self.inicio+6*self.gesto+5*self.descp+2*self.fatiga,2*self.fatiga+2*self.inicio+7*self.gesto+5*self.descp],
     [2*self.inicio+7*self.gesto+6*self.descp+2*self.fatiga,2*self.fatiga+2*self.inicio+8*self.gesto+6*self.descp],
     [2*self.inicio+8*self.gesto+7*self.descp+2*self.fatiga,2*self.fatiga+2*self.inicio+9*self.gesto+7*self.descp],
     [2*self.inicio+9*self.gesto+8*self.descp+2*self.fatiga,2*self.fatiga+2*self.inicio+10*self.gesto+8*self.descp]]


    # Lista de niveles soportados
    self.niveles = [1, 2, 3]

    # Relaciona niveles con las estructuras que contienen su forma
    self.formato_nivel = {1: self.nivel1g, 2: self.nivel2g, 3: self.nivel3g}
    self.formato_descanso = {1: self.nivel1d, 2: self.nivel2d, 3: None}

    # Tamaño de ventana en milisegundos
    self.tam = 200
    # Frecuencia de muestreo
    self.fs = 200  # Hz
    # Overlap entre las muestras
    self.over = 0.5

  def graficar(senal):
    """
    Plot the EMG signal.

    Args:
    senal (DataFrame): EMG signal data.

    Returns:
    None
    """
    # Crear una figura y ejes
    fig, ax = plt.subplots()

    # Graficar las listas en la misma gráfica
    ax.plot(senal[0], label='Lista 1')
    ax.plot(senal[1], label='Lista 2')
    ax.plot(senal[2], label='Lista 3')
    ax.plot(senal[3], label='Lista 4')
    ax.plot(senal[4], label='Lista 5')
    ax.plot(senal[5], label='Lista 6')
    ax.plot(senal[6], label='Lista 7')
    ax.plot(senal[7], label='Lista 8')

    # Agregar leyenda
    ax.legend()

    # Mostrar la gráfica
    plt.show()

  def separacion(dataframe):
    """Transpose the dataframe.

    Args:
    dataframe (DataFrame): EMG signal data.

    Returns:
    DataFrame: Transposed EMG signal data.
    """
    dataframe_invertido = dataframe.T
    return dataframe_invertido

  def simplificarGesto(self,senal):
    """
    Simplifies the given signal by performing a separation operation.

    Parameters:
    - senal: The input signal to be simplified.

    Returns:
    - sep: The simplified signal after separation.
    """
    sep = self.separacion(senal)
    return sep

  def segmentar(self, gesto):
    """
    Segments a signal into overlapping windows.

    Args:
    gesture (DataFrame): The signal to be segmented..

    Returns:
    list:  A list of overlapping windows..

    """
    muestras_por_ventana = int(self.tam * self.fs / 1000)
    # Superposición del 50%
    sup = int(muestras_por_ventana * self.over)
    # Número total de ventanas considerando el 50% de superposición
    total_ventanas = int((len(gesto) - muestras_por_ventana) / sup) + 1

    # DataFrame vacío
    ventanas=[]
    yObj=[]

    inicio = 0
    fin = inicio + muestras_por_ventana

    # Iterar a través de las señales y extraer las ventanas
    for j in range(total_ventanas):

      if fin <= len(gesto):
        ventana = gesto.iloc[inicio:fin]
        ventanas.append(ventana)
        inicio=inicio+sup
        fin=fin+sup

    return ventanas

  def procesarVentanas(self,gesto):
    """Process the segmented EMG signal windows.

    Args:
    gesto (DataFrame): EMG signal data.

    Returns:
    list: List of processed EMG signal windows.
    """
    ventanas=self.segmentar(gesto)
    simps=[]
    for i in ventanas:
      aux=self.simplificarGesto(i)
      simps.append(aux)

    return(simps)

  def buscarGestos(self,emg,lv):
    """Find gestures in the EMG signal.

    Args:
    emg (DataFrame): EMG signal data.
    lv (int): Level of gestures to search for.

    Returns:
    list: List of segmented and processed EMG signal windows for the gestures.
    """

    stc=self.formatoNivel[lv]

    gestos=[]

    for i in range(10):

      seg= emg.iloc[(stc[i][0])*self.fs:(stc[i][1])*self.fs]
      gestos=gestos+self.procesarVentanas(seg)

    return(gestos)

  def buscarDescanso(self,emg,lv):
    """Find rest periods in the EMG signal.

    Args:
    emg (DataFrame): EMG signal data.
    lv (int): Level of rest periods to search for.

    Returns:
    list: List of segmented and processed EMG signal windows for the rest periods.
    """

    stc=self.formatoDescanso[lv]
    desc=[]

    if(lv!=3):

      seg= emg.iloc[(stc[0][0])*self.fs:(stc[0][1])*self.fs]
      desc=desc+self.procesarVentanas(seg)

    return(desc)


  def cargarArchvivo(path):
      """Load the EMG signal data from a file.

    Args:
      path (str): Path to the file.

    Returns:
      DataFrame: EMG signal data.
    """
      try:
          mat= scipy.io.loadmat(path)
          signal=mat['emgSignalOut']
          return pd.DataFrame(signal)
      except FileNotFoundError:
          print(f"La ruta '{path}' no existe. Ignorando este archivo.")
          return None

  def etiquetar(self, gesto, num):
    """Label the gestures.

    Args:
      gesto (str): Gesture name.
      num (int): Number of times the gesture appears.

    Returns:
      list: List of gesture labels.
    """
    return [self.GestosInt[gesto] for i in range(num)]

  def generar(self,data):
    """Generate the dataset.

    Args:
      data (list): List of tuples containing file paths, gesture names, and levels.

    Returns:
      tuple: Tuple containing the input data (x) and the labels (y).
    """
    x=[]
    y=[]

    for i in data:
      file=self.cargarArchvivo(i[0])
      if file is not None:
        g=self.buscarGestos(file,i[2])
        x=x+g
        y=y+self.etiquetar(i[1], len(g))
        d=self.buscarDescanso(file,i[2])
        x=x+d
        y=y+self.etiquetar("Desc", len(d))

    return x, y


  def getSignalsxUser(self,user):
    """Get the file paths for a user's EMG signals.

    Args:
      user (str): User name.

    Returns:
      list: List of file paths.
    """
    
    direc=[]
    for i in self.Gestos:
      if(i!="Desc"):
        direc=direc+self.getSignalsxMov(user,i)
    return direc

  def getSignalsxMov(self, path, user,mov):
    """Get the file paths for a user's EMG signals for a specific gesture.

    Args:
      user (str): User name.
      mov (str): Gesture name.

    Returns:
      list: List of file paths.
    """
    full=path+"/"+user+"/"+mov
    direc=[]
    for i in self.niveles:
      direc.append(self.getSignalsxLevel(user,mov,i))
    return direc

  def getSignalsxLevel(path,user,mov,lv):
    """Get the file path for a user's EMG signal for a specific gesture and level.

    Args:
      user (str): User name.
      mov (str): Gesture name.
      lv (int): Level.

    Returns:
      str: File path.
    """

    file="emg_"+mov+".mat"
    full=path+"/"+user+"/"+mov+"/"+str(lv)+"/"+file
    #print(full)
    return (full,mov,lv)

  #Localiza usuario
  def buscarUser(tamUsers):
    """Find the user names.

    Args:
      tamUsers (int): The number of users to find.

    Returns:
      list: List of user names.
    """
    return ["user"+str(i) for i in range(tamUsers)]

  def crearDataset(self):
    """Create the dataset.

    Returns:
      tuple: Tuple containing the input data (x) and the labels (y).
    """
    usuarios=self.buscarUser()

    x=[]
    y=[]

    for i in usuarios:
      rutas=self.getSignalsxUser(i)
      aux,auy=self.generar(rutas)
      x=x+aux
      y=y+auy

    return(x,y)
