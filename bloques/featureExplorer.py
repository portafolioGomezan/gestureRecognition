# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import pandas as pd

"""codigo relacionado con la grafica de señales"""
import matplotlib.pyplot as plt
from matplotlib import pyplot

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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import davies_bouldin_score

from mrmr import mrmr_classif
from scipy.stats import ttest_ind


class featureExplorer:
    """
    A class that represents a feature explorer.

    Attributes:
    GestosInt (dict): A dictionary that maps gesture names to their corresponding integer values.
    IntGestos (dict): A dictionary that maps integer values to their corresponding gesture names.
    nombresCarac (list): A list of feature names.
    nombresFull (list): A list of feature names with index numbers.
    nombresFullIndex (range): A range of index numbers for feature names.
    numClases (int): The number of classes.

    Methods:
    __init__(): Initializes the featureExplorer object.
    plot_box(data, titulo): Plots a boxplot for the given data.
    plot_matrix(data, titulo): Plots a matrix of data with specified title.
    pca_analysis(x, tam): Performs PCA analysis on the given data.
    forest_test(x, y): Performs feature selection using ExtraTreesClassifier.
    mrmr_test(x, y): Performs feature selection using MRMR algorithm.
    t_test(vF, vL, nClass): Performs t-test for feature selection.
    DaviesBouldinIndex(vF, vL, nFeatures): Calculates the Davies Bouldin Index for feature selection.
    """

    def __init__(self):
        """
        Initializes a FeatureExplorer object.

        The FeatureExplorer class is used for exploring features related to gestures.
        It initializes various dictionaries and lists used for mapping and storing information.

        Attributes:
        - GestosInt (dict): A dictionary mapping gesture names to integer values.
        - IntGestos (dict): A dictionary mapping integer values to gesture names.
        - nombresCarac (list): A list of feature names.
        - nombresFull (list): A list of feature names combined with index values.
        - nombresFullIndex (range): A range of index values for the feature names.
        - IntGestosFull (dict): A dictionary mapping index values to feature names.
        - numClases (int): The number of classes.

        """
        self.GestosInt = {"RMS":0,"STD":1,"Varianza":2,"MAV":3,"WL":4,"Promedio":5,"ZC":6,"kurtosis":7,"skewness":8, "iEMG":9, "SSC":10,
                            "WAMP":11,"MAS":12, "MP":13, "MDF":14, "MNF":15}
        self.IntGestos = {0:"RMS",1:"STD",2:"Varianza",3:"MAV",4:"WL",5:"Promedio",6:"ZC",7:"kurtosis",8:"skewness", 9:"iEMG",
                            10:"SSC",11: "WAMP",12:"MAS",13: "MP",14: "MDF",15: "MNF"}
        self.nombresCarac = ["RMS","STD","Varianza","MAV","WL","Promedio","ZC","kurtosis","skewness", "iEMG", "SSC", "WAMP",
                                "MAS", "MP", "MDF", "MNF"]
        self.nombresFull = [j+str(i) for i in range(8) for j in self.nombresCarac]
        self.nombresFullIndex = range(8 * len(self.nombresCarac))
        self.IntGestosFull = dict(zip(self.nombresFullIndex, self.nombresFull))
        self.numClases = 6


    """## Funciones importantes"""

    def plot_box(self,data, titulo):
        """
        Plots a boxplot for the given data.

        Parameters:
        data (list): A list of lists or arrays containing the data for each boxplot.
        titulo (str): The title of the plot.

        Returns:
        None
        """
        fig, ax = plt.subplots()
        names = ["MRMR", "RanForest", "t test", "DB"]
        ax.set_xticklabels(names)

        plt.xlabel("Pruebas", size=14)
        plt.ylabel("Resultados", size=14)

        # Creating plot
        ax.boxplot(data)
        plt.title(titulo)

        # Show plot
        plt.show()



    def plot_matrix(self, data, titulo):
        """
        Plots a matrix of data with specified title.

        Args:
            data (list): A list of 128 features.
            titulo (str): The title of the plot.

        Raises:
            ValueError: If the length of the data is not equal to 128.

        Returns:
            None
        """

        if len(data) != 128:
            raise ValueError("Es necesario recibir 128 caracteristicas")

        matrix = np.array(data).reshape(8, 16)

        newNames=self.nombresCarac.copy()
        newNames[7]="skw"
        newNames[8]="kts"

        pyplot.figure(figsize=(15,10))

        pyplot.xlabel("Caracteristicas",size = 14)
        pyplot.ylabel("Canales",size= 14)
        pyplot.xticks(np.arange(len(self.nombresCarac)), newNames)

        pyplot.title(titulo,size=28)

        grid_pointsy = np.arange(0.5,8.5,1)
        grid_pointsx = np.arange(0.5,16.5,1)
        custom_grid_points=np.array(np.meshgrid(grid_pointsx, grid_pointsy)).T.reshape(-1, 2)


        # Dibujar la malla personalizada
        for point in custom_grid_points:
            pyplot.plot(point[0], point[1], 'o', markersize=3, color='gray')

        pyplot.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        pyplot.gca().set_xticks(np.arange(-0.5, 16, 1), minor=True)
        pyplot.gca().set_yticks(np.arange(-0.5, 8, 1), minor=True)

        imagen=pyplot.imshow(matrix, cmap="jet")

        pyplot.colorbar(imagen,shrink=0.5)
        pyplot.show()


    """##PCA

    Se evaluan los componenetes y extraen aquellos que conservan la mayor cantidad de variabilidad.
    """

    def pca_analysis(self,x, tam):
        """
        Performs PCA analysis on the given data.

        Args:
            x (array-like): The input data.
            tam (int): The number of components to keep.

        Returns:
            None
        """
        pcam = PCA(n_components=tam)
        pcam.fit(x)

        explained_variance = pcam.explained_variance_ratio_
        selected_features = np.argsort(explained_variance)

        print(explained_variance)
        print(selected_features)

        fig, ax = plt.subplots()
        ax.plot(np.cumsum(pcam.explained_variance_ratio_), label = 'ExplainedVariance')
        ax.legend(loc = 'upper right')
        plt.grid(True)
        plt.xlabel('Component number')
        plt.ylabel('Variance')
        plt.title('PCA')
        plt.show()



        """## Arbol aleatorio

        """

    def forest_test(self,x,y):
        """
        Performs feature selection using ExtraTreesClassifier.

        Args:
            x (array-like): The input data.
            y (array-like): The target labels.

        Returns:
            forestResults (pd.Series): The feature importances obtained from ExtraTreesClassifier.
        """
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(x, y)
        print(clf.feature_importances_)

        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(x)
        print(X_new.shape)

        forestResults=pd.Series(clf.feature_importances_).copy()

        self.plot_matrix(clf.feature_importances_,"Importancia según bosques aleatorios")

        return forestResults


        #MRMR

    def mrmr_test(self, x, y):
        """
        Performs feature selection using MRMR algorithm.

        Args:
            x (array-like): The input data.
            y (array-like): The target labels.

        Returns:
            None
        """

        if not isinstance(x, pd.DataFrame):
            x=pd.DataFrame(x)

        if not isinstance(y, pd.DataFrame):
            y=pd.DataFrame(y)


        selected_features = mrmr_classif(X=x, y=y, K=len(self.nombresCarac),return_scores=True)
        print(selected_features[1])


        MRMRResults=pd.Series([selected_features[1][i] for i in range(len(selected_features[1]))])
        self.plot_matrix(selected_features[1],"Análisis MRMR")

        return(MRMRResults)


    def t_test(self,vF, vL):
        """
        Performs t-test for feature selection.

        Args:
            vF (pd.DataFrame): The feature data.
            vL (pd.DataFrame): The label data.

        Returns:
            vsF (pd.Series): The p-values obtained from t-test.
        """

        if not isinstance(vF, pd.DataFrame):
            vF=pd.DataFrame(vF)
            vF.columns=self.nombresFull

        if not isinstance(vL, pd.DataFrame):
            vL=pd.DataFrame(vL)

        nClass=self.numClases
        alpha = 0.01
        res=[]

        for idxC1 in range(nClass):
            for idxC2 in range(idxC1+1 , nClass ):

                data=[]
                for idxCF in self.nombresFull:

                    data1=vF[vL[0]==idxC1]
                    data2=vF[vL[0]==idxC2]
                    ax = data1[idxCF]  # clase idxC1 caracteristica idxCF
                    ay = data2[idxCF]  # clase idxC2 caracteristica idxCF

                        # Calculate p-value using t-test
                    _, p_value = ttest_ind(ax, ay, equal_var=False)
                    data.append(p_value)

                res.append(data)

        vsF=pd.DataFrame(res).mean()
        self.plot_matrix(vsF,"Análisis t test")
        return vsF



    """## Indice de Davies Bouldin"""

    def DaviesBouldinIndex(self, vF, vL):
        """
        Calculates the Davies Bouldin Index for feature selection.

        Args:
            vF (pd.DataFrame): The feature data.
            vL (pd.DataFrame): The label data.
            nFeatures (int): The number of features.

        Returns:
            None
        """
        if not isinstance(vF, pd.DataFrame):
            vF=pd.DataFrame(vF)
            vF.columns=self.nombresFull

        if not isinstance(vL, pd.DataFrame):
            vL=pd.DataFrame(vL)

        nFeatures = len(self.nombresFull)

        DB = np.zeros(nFeatures)

        for idxCF in range(nFeatures):
            ax = np.array(vF[self.nombresFull[idxCF]])
            ax=np.reshape(ax,(-1,1))
            DB[idxCF]= davies_bouldin_score(ax, vL[0])

        self.plot_matrix(DB,"Davies Bouldin")

        return DB

