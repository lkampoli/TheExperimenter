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
from sklearn_som.som import SOM
from minisom import MiniSom  

# project imports
from utils.logger_config import Logger


class _SOM_:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    SOM algorithm.

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
        Run the TPOTRegressor algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """
        
        XYZ = train_data_x[['Cx','Cy','Cz']]
        X   = train_data_x[['DELTA_Rxx', 'DELTA_Rxy', 'DELTA_Ryy']]
        X   = X.values
    
        # https://python.hotexamples.com/examples/som/SOM/get_weights/python-som-get_weights-method-examples.html
        # https://github.com/FlorentF9/SOMperf
        # https://github.com/andremsouza/python-som
        # https://sklearn-som.readthedocs.io
        # class sklearn_som.som.SOM(m=3, n=3, dim=3, lr=1, sigma=1, max_iter=3000, **kwargs)
                
        # Initialization and training
        som_shape = (10, 10)

        algo = MiniSom(som_shape[0], som_shape[1], X.shape[1], sigma=.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=73)
        print('SOM clustering algorithm instantiated')

        t0 = time()
        algo.train_batch(X, 10000, verbose=True)
        t1 = time()

        max_iter = 1000
        q_error = []
        t_error = []
        
        for i in range(max_iter):
            rand_i = np.random.randint(len(X))
            algo.update(X[rand_i], algo.winner(X[rand_i]), i, max_iter)
            q_error.append(algo.quantization_error(X))
            t_error.append(algo.topographic_error(X))
        
        plt.plot(np.arange(max_iter), q_error, label='quantization error')
        plt.plot(np.arange(max_iter), t_error, label='topographic error')
        plt.ylabel('Quantization error')
        plt.xlabel('Iteration index')
        plt.legend()
        # save figs
        algo_error_file_path = os.path.join(save_dir, 'som_errors.eps')
        plt.savefig(algo_error_file_path)
        plt.close()

        # each neuron represents a cluster
        winner_coordinates = np.array([algo.winner(x) for x in X]).T
        # with np.ravel_multi_index we convert the bidimensional
        # coordinates to a monodimensional index
        cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
        
        plt.figure(figsize=(30,15))
        
        # plotting the clusters using the first 2 dimentions of the data
        for c in np.unique(cluster_index):
            plt.scatter(X[cluster_index == c, 0],
                        X[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
        
        # plotting centroids
        for centroid in algo.get_weights():
            plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                        s=80, linewidths=35, color='k', label='centroid')
        plt.legend()
        cluster_centroids_file_path = os.path.join(save_dir, 'som_centroids.eps')
        plt.savefig(cluster_centroids_file_path)
        plt.close()

        try:
            if hasattr(algo, "labels_"):
                labels = algo.labels_.astype(np.int32)
            else:
                labels = algo.predict(X)
        except:
            labels = np.ravel_multi_index(winner_coordinates, som_shape)
            #print("Labels not available")

        try:
            centroids = algo.cluster_centers_
        except:
            centroids = algo.get_weights()
            #print("Centroids not available")

        plt.figure(figsize=(30, 15))
        plt.title("SOM", size=18)

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
        cluster_algo_file_path = os.path.join(save_dir, 'SOM.eps')
        plt.savefig(cluster_algo_file_path)

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
        Logger.print("\nFinished SOM clustering run")
        return results

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<SOM>"
