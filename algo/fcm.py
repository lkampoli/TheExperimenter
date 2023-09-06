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
from fcmeans import FCM

# project imports
from utils.logger_config import Logger


class _FCM_:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    FCM algorithm.

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
        Run the FCM algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """

        XYZ = train_data_x[['Cx','Cy','Cz']]
        X   = train_data_x[['DELTA_Rxx', 'DELTA_Rxy', 'DELTA_Ryy']]

        algo = FCM(n_clusters=n_clusters, random_state=73)
        print('FCM clustering algorithm instantiated')

        t0 = time()
        algo.fit(X)
        t1 = time()

        if hasattr(algo, "labels_"):
            labels = algo.labels_.astype(np.int32)
        else:
            labels = algo.predict(X)

        try:
            centroids = algo.cluster_centers
        except:
            print("Centroids not available")

        plt.figure(figsize=(30, 15))
        plt.title("FCM", size=18)

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
        cluster_algo_file_path = os.path.join(save_dir, 'FCM.eps')
        plt.savefig(cluster_algo_file_path)
        plt.close()

        # prepare DF for results
        results = pd.DataFrame()

        for test in range(run_times):
            Logger.print("Clustering run {}/{}".format(test + 1, run_times))

            try:
                results["labels"] = labels
            except:
                print("No labels available")

            try:
                results["centroids"] = centroids.ravel()
            except:
                print("No centroids available")

        # Logger.print and save scoring results of all runs
        Logger.print("\nFinished FCM clustering run")
        return results

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<FCM>"
