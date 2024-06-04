
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
  x = np.load("caracterisricas.npy")
  y = np.load("etiquetas.npy")

  tam = x.shape[1]

  # Perform PCA analysis of the initial data
  fe.pca_analysis(x, tam)

  # Analyze the variances of the features from data
  variances = np.var(x, axis=0)
  column_labels = fe.nombresCarac
  fe.plot_matrix(variances, "Varianza explicada")

  # Perform forest test to determine the best features of the initial data
  importance = fe.forest_test(x, y)
  prom = importance.mean()
  serie_filtrada = importance[importance >= prom]
  print("Los mejores gestos son: ")
  forestBest = [fe.IntGestosFull[i] for i in serie_filtrada.index]
  print(len(forestBest))

  forestResults = pd.Series(importance).copy()

  # Perform MRMR test to determine the best features of the initial data
  selected_features = fe.mrmr_test(x, y)
  prom = selected_features[1].mean()
  serie_filtrada = selected_features[1][selected_features[1] >= prom]
  print("Los mejores gestos son: ")
  MRMRbest = [i for i in serie_filtrada.index]
  print(len(MRMRbest))

  MRMRResults = pd.Series([selected_features[1][i] for i in range(len(selected_features[1]))])
  print(MRMRResults)

  # Perform t-test to determine the best features of the initial data
  vsF = fe.t_test(x, y, fe.numClases)
  tResults = vsF * 4

  print(vsF.max())

  fe.plot_box(vsF, "Resultados t test")
  fe.plot_box(tResults, "Resultados t test")
  fe.plot_matrix(vsF, "Pruebas T")

  # Perform Davies-Bouldin index to determine the best features of the initial data
  DB = fe.DaviesBouldinIndex(x, y, fe.numClases, len(fe.nombresFull))
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

  fe.plot_box(scaledMetric, "Diagrama de bigotes métricas")

  score = np.sum(scaledMetric, axis=1)
  print(score.shape)
  pf = pd.DataFrame(score, columns=["Puntaje"])
  pf.head()

  pfs = pf.sort_values(by="Puntaje", ascending=False).copy().reset_index().reset_index()
  pfs.columns = ["Posicion", "Caracteristica", "Puntaje"]
  pfs.head()

  resFinal = np.array(pfs["Caracteristica"])
  print(resFinal.shape)
  np.save("Pnc.npy", resFinal)

  """
  We join the results of the 4 test made previously and plot the results.
  """
  limit = 60
  sclf = pfs.copy()
  sclf.iloc[limit:, 2] = 0
  sclfCar = sclf.sort_values(by="Caracteristica")

  sclfCar["best"] = sclfCar["Puntaje"].apply(lambda x: 1 if x > 0 else 0)
  sclfCar.head()

  fe.plot_matrix(sclfCar["best"], "Mejores caracteristicas")

  fe.plot_matrix(sclfCar["Puntaje"], "Mejores caracteristicas")

  sel = ["RMS", "STD", "Varianza", "MAV", "WL", "MP", "MDF", "MNF"]
  nombresSel = [j + str(i) for i in range(8) for j in sel]

  dtSel = x[nombresSel]

  print(dtSel.shape)
  caracFull = pd.concat([dtSel, y], axis=1)
  caracFull.to_csv('caracFull.csv', index=False, header=True, encoding='utf-8')
  caracFull.head()

  x.head()

  print("Las mejores caracteristicas son:")
  print(MRMRbest)
  print(forestBest)

  conj1 = set(MRMRbest)
  conj2 = set(forestBest)
  best = conj1 & conj2

  bestDF = x.loc[:, list(best)]
  bestDF.to_csv('best.csv', index=False, header=True, encoding='utf-8')

  print(bestDF.shape)

  totalBestDF = pd.concat([bestDF, y], axis=1)
  totalBestDF.to_csv('totalBest.csv', index=False, header=True, encoding='utf-8')
  totalBestDF.head()

  