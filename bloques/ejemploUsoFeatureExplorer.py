
import numpy as np
import pandas as pd
import featureExplorer as fe

from sklearn.preprocessing  import MinMaxScaler



def main():
  """
  This is the main function that performs various tests and analysis on the data.
  It performs PCA analysis, analyzes variances of features, performs forest test,
  MRMR test, t-test, Davies-Bouldin index, and joins the results of the tests.
  It also scales the metrics, calculates scores, and saves the results to files.
  """
  x = np.load("xtrainC.npy")
  y = np.load("y.npy")

  tam = x.shape[1]
  print(x.shape)
  print(y.shape)

  feature_explorer = fe.featureExplorer()

  # Perform PCA analysis of the initial data
  feature_explorer.pca_analysis(x, tam)

  # Analyze the variances of the features from data
  variances = np.var(x, axis=0)
  column_labels = feature_explorer.nombresCarac
  feature_explorer.plot_matrix(variances, "Varianza explicada")

  # Perform forest test to determine the best features of the initial data
  importance = feature_explorer.forest_test(x, y)
  forestResults = pd.Series(importance).copy()

  # Perform MRMR test to determine the best features of the initial data
  MRMRResults = feature_explorer.mrmr_test(x, y)

  # Perform t-test to determine the best features of the initial data
  vsF = feature_explorer.t_test(x, y)
  tResults = vsF * 4

  # Perform Davies-Bouldin index to determine the best features of the initial data
  DB = feature_explorer.DaviesBouldinIndex(x, y)
  DBResults = pd.Series(DB).apply(lambda x: 1 / x)

  # Join the results of the tests
  resultadosMeticas = {"MRMR": MRMRResults,
             "Random Forest": forestResults,
             "T test": tResults,
             "Davies-Bouldin": DBResults}

  metrics = pd.DataFrame(resultadosMeticas)
  metrics.head()

  scaler = MinMaxScaler()
  scaler.fit(metrics)
  scaledMetric = scaler.transform(metrics)
  print(scaledMetric)

  feature_explorer.plot_box(scaledMetric, "Diagrama de bigotes métricas")

  
if __name__ == "__main__":
  main()