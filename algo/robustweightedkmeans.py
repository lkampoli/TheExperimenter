# library imports
import os
from time import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn_extra.robust import RobustWeightedKMeans

# project imports
from utils.logger_config import Logger


class ROBUSTWEIGHTEDKMEANS:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    RobustWeightedKMeans algorithm.

    Input dataset is used to train and test the model
    multiple times (named as run_times), to gain statistical
    insight on the performance.
    """

    def __init__(self):
        pass

    @staticmethod
    def run_and_analyze(run_times: int,
                        n_clusters: int,
                        train_data_x: pd.DataFrame,
                        train_data_y: pd.DataFrame,
                        test_data_x: pd.DataFrame,
                        test_data_y: pd.DataFrame,
                        #generations: int,
                        #population_size: int,
                        #k_fold: int,
                        performance_metric,
                        save_dir: str,
                        n_jobs: int = -1):
        """
        Run the RobustWeightedKMeans algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """

        XYZ = train_data_x[['Cx','Cy','Cz']]
        X   = train_data_x[['DELTA_Rxx', 'DELTA_Rxy', 'DELTA_Ryy']]

        algo = RobustWeightedKMeans(n_clusters=n_clusters, weighting="mom", max_iter=100, random_state=73)
        print('clustering algorithm instantiated')

        t0 = time()
        algo.fit(X)
        t1 = time()

        if hasattr(algo, "labels_"):
            labels = algo.labels_.astype(np.int32)
        else:
            labels = algo.predict(X)

        centroids = algo.cluster_centers_

        plt.figure(figsize=(30, 15))
        plt.title("RobustWeightedKMeans", size=18)

        try:
            plt.scatter(XYZ['Cx'], XYZ['Cy'], s=100, c=labels)
        except:
            print("Cx, Cy, Cz not found")
    
        try:
            plt.scatter(XYZ['X'], XYZ['Y'], s=100, c=labels)
        except:
            print("X, Y, Z not found")
            
        try:
            plt.scatter(XYZ['Points:0'], XYZ['Points:1'], s=100, c=labels)
        except:
            print("Points:0, Points:1, Points:2 not found")
        
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        cluster_algo_file_path = os.path.join(save_dir, 'RobustWeightedKMeans.eps')
        plt.savefig(cluster_algo_file_path)
        plt.close()

        # prepare DF for results
        results = pd.DataFrame()

        for test in range(run_times):
            Logger.print("Clustering run {}/{}".format(test + 1, run_times))

            try:
                #results.at[test, "labels"] = labels
                results["labels"] = labels
            except:
                print("No labels available")

            try:
                #results.at[test, "centroids"] = centroids
                results["centroids"] = centroids.ravel()
            except:
                print("No centroids available")

        # Logger.print and save scoring results of all runs
        Logger.print("\nFinished RobustWeightedKMeans clustering run")
        return results

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<RobustWeightedKMeans>"
