# library imports
import os
from time import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
import fastcluster

# project imports
from utils.logger_config import Logger


class AGGLOMERATIVECLUSTERING:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    AgglomerativeClustering algorithm.

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
        Run the AgglomerativeClustering algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """

        XYZ = train_data_x[['Cx','Cy','Cz']]
        X   = train_data_x[['DELTA_Rxx', 'DELTA_Rxy', 'DELTA_Ryy']]

#        # fastcluster # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        classes = ['g'] * 50 + ['r'] * 50 + ['c'] * 50
#        N = len(X)
#        def plot_with_labels(Z, num_clust):
#            # plot the dendrogram
#            threshold = Z[-num_clust + 1, 2]
#            dg = dendrogram(Z, no_labels = True, color_threshold = threshold)
#            # plot color bars under the leaves
#            color = [classes[k] for k in dg['leaves']]
#            b = .1 * Z[-1, 2]
#            plt.bar(np.arange(N) * 10, np.ones(N) * b, bottom = -b, width = 10,
#                    color = color, edgecolor = 'none')
#            plt.gca().set_ylim((-b, None))
#            fastcluster_algo_file_path = os.path.join(save_dir, 'fastcluster_dendrogram.eps')
#            plt.savefig(fastcluster_algo_file_path)
#            plt.close()
#        
#        Z = fastcluster.linkage(X, method = 'single')
#        plot_with_labels(Z, 2)
#        
#        D = pdist(X, metric = 'cityblock')
#        Z = fastcluster.linkage(D, method = 'weighted')
#        plot_with_labels(Z, 3)
#        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        algo = AgglomerativeClustering(n_clusters=n_clusters)
        print('AgglomerativeClustering clustering algorithm instantiated')

        t0 = time()
        algo.fit(X)
        t1 = time()

        if hasattr(algo, "labels_"):
            labels = algo.labels_.astype(np.int32)
        else:
            labels = algo.predict(X)
        Logger.print("Labels: {}, size: {}".format(np.unique(labels), len(labels)))

        try:
            centroids = algo.cluster_centers_
        except:
            print("Centroids not available")

        plt.figure(figsize=(30, 15))
        plt.title("AgglomerativeClustering", size=18)

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
        cluster_algo_file_path = os.path.join(save_dir, 'AgglomerativeClustering.eps')
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
        Logger.print("\nFinished AgglomerativeClustering clustering run")
        return results

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<AgglomerativeClustering>"
