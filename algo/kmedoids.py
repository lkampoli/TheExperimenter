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
from sklearn_extra.cluster import KMedoids

# project imports
from utils.logger_config import Logger


class KMEDOIDS:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    KMedoids algorithm.

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

        algo = KMedoids(n_clusters=n_clusters, random_state=73, metric="euclidean")
        #algo = KMedoids(n_clusters=n_clusters, random_state=73, metric="manhattan")
        #algo = KMedoids(n_clusters=n_clusters, random_state=73, metric="cosine")

        t0 = time()
        algo.fit(train_data_x)
        t1 = time()

        if hasattr(algo, "labels_"):
            labels = algo.labels_.astype(np.int32)
        else:
            labels = algo.predict(train_data_x)

        centroids = algo.cluster_centers_

        plt.figure(figsize=(30, 15))
        plt.title("KMedoids", size=18)

        #plt.savefig('clustering_algorithms_'+str(point_percentage)+'%_'+str(clusters)+'_clusters_'+'scaled_'+str(scale)+'.pdf')    
    
        #names = list(data.columns.values.tolist())
        #np.savetxt('labels_'+str(names)+'.txt', y_pred, fmt='%i')

        try:
            plt.scatter(train_data_x['Cx'], train_data_x['Cy'], s=100, c=labels)
        except:
            print("Cx, Cy, Cz not found")
    
        try:
            plt.scatter(train_data_x['X'], train_data_x['Y'], s=100, c=labels)
        except:
            print("X, Y, Z not found")
            
        try:
            plt.scatter(train_data_x['Points:0'], train_data_x['Points:1'], s=100, c=labels)
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
        cluster_algo_file_path = os.path.join(save_dir, 'KMEDOIDS.eps')
        plt.savefig(cluster_algo_file_path)

#        # const
#        tpot_model_file_path = os.path.join(save_dir, 'current_tpot_pipeline.py')
#        tpot_object_file_path = os.path.join(save_dir, 'current_tpot_pipeline_as_object')
#        # prepare DF for results
#        best_model = None
#        results = pd.DataFrame()
#        current_best_performance_score = 99999
#
#        for test in range(run_times):
#            Logger.print("TPOT run {}/{}".format(test + 1, run_times))
#            model = TPOTRegressor(generations=generations,
#                                  population_size=population_size,
#                                  cv=KFold(n_splits=k_fold),
#                                  scoring=performance_metric,
#                                  verbosity=2,
#                                  n_jobs=n_jobs)
#            model.fit(train_data_x, train_data_y)
#            pred = model.predict(test_data_x)
#
#            # store test scores
#            try:
#                # we assume this is just a function
#                performance_score = performance_metric(test_data_y, pred)
#            except:
#                # maybe it is a scorer wrapper of a function and we want to overcome it
#                performance_score = performance_metric._score_func(test_data_y, pred)
#
#            results.at[test, "performance_score"] = performance_score
#            results.at[test, "mae"] = mean_absolute_error(test_data_y, pred)
#            results.at[test, "mse"] = mean_squared_error(test_data_y, pred)
#            results.at[test, "r2"] = r2_score(test_data_y, pred)
#            results.at[test, "t_test_p_value"] = stats.ttest_ind(test_data_y, pred)[1]
#
#            # store exported pipeline
#            model.export(tpot_model_file_path)
#            pipeline = TPOTresultsExtractor.process_file(tpot_model_file_path)
#            results.at[test, 'pipeline'] = pipeline
#
#            # update best mae score and model
#            if performance_score < current_best_performance_score:
#                best_model = model
#                current_best_performance_score = performance_score
#
#        # remove unnecessary file
#        os.remove(tpot_model_file_path)
#
        # Logger.print and save scoring results of all runs
        Logger.print("\nFinished KMedoids clustering run")
        #[Logger.print("{}: {:.3}+-{:.3}".format(score, results[score].mean(), results[score].std())) for score in results.keys() if score != "pipeline"]
        #return centroids, labels
        return labels

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<KMedoids>"
