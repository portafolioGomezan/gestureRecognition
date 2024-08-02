# Main code
# ...

import numpy as np
import signalProcessor as sp
import pandas as pd

def main():
  """
  Main function for the example usage of the SignalProcessor class.
  """
  # Se crea un dataset con la función 'crearDataset()
  signalProcessor = sp.SignalProcessor()
 
  signalProcessor.set_path(r"C:\Users\aulasingenieria\Desktop\signalTest")

  #Necesitamos conocer el número de sujetos
  x, y = signalProcessor.crearDataset(1)
  

  # Se imprimen detalles del dataset 'x'
  print(type(x))  # Muestra el tipo de datos de 'x' (probablemente una lista)
  print(len(x))   # Muestra la longitud de 'x' (cantidad de muestras)
  print(type(x[0]))  # Muestra el tipo de datos del primer elemento de 'x' (probablemente un DataFrame de pandas)
  print(x[0].head())  # Muestra las primeras filas del primer DataFrame en 'x'

  # Se imprimen detalles del dataset 'y'
  print(type(y))  # Muestra el tipo de datos de 'y' (probablemente una lista)
  print(len(y))   # Muestra la longitud de 'y' (cantidad de etiquetas)
  print(type(y[0]))  # Muestra el tipo de datos del primer elemento de 'y' (probablemente un entero o una etiqueta categórica)

  # Se convierte 'y' en un DataFrame de pandas y se muestran sus primeras filas
  ydf = pd.DataFrame(y)
  print(ydf.head())

  # Se cuenta el número de veces que aparece cada gesto en 'y'
  for i in range(len(signalProcessor.Gestos)):
    count = ydf[0].value_counts()[i]
    print("El número de veces de", i, " y es: ", count)

  # Se convierte 'x' y 'y' a arreglos de numpy y se imprimen sus formas
  xarray = np.array(x)
  print(xarray.shape)
  np.save("x.npy",xarray)

  yarray = np.array(y)
  print(yarray.shape)
  np.save("y.npy",yarray,)


if __name__ == "__main__":
  main()