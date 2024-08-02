
import numpy as np
import pandas as pd
import featureSelector as fs

def main():
  """
  This is the main function that performs the following steps:
  1. Loads the 'x' and 'y' data from files.
  2. Reshapes the 'x' data.
  3. Converts the reshaped 'x' data into a DataFrame.
  4. Prints the length of the DataFrame.
  5. Prints the first few rows of the DataFrame.
  6. Caracterizes the features of the dataset using the 'fs' featureSelector.
  7. Prints the length of the caracterized features.

  Returns:
  None
  """
  x = np.load('x.npy')
  forma = x.shape

  xCalc = x.reshape((forma[0]*forma[1], forma[2]))
  xdf = pd.DataFrame(xCalc)
  print(xdf.shape)

  # Get the features of the dataset calculated by the featureSelector
  featureSelector = fs.FeatureSelector()
  xtrainC = featureSelector.caracterizar(xdf)
  print(len(xtrainC))

  xn=np.array(xtrainC)
  new=xn.reshape((forma[0],8*len(featureSelector.nombresCarac)))
  xtrainC=pd.DataFrame(new)
  xtrainC = xtrainC.rename(columns=dict(zip(xtrainC.columns, featureSelector.nombresFull)))

  array = xtrainC.to_numpy()
  np.save("xtrainC.npy",array)




if __name__ == "__main__":
  main()