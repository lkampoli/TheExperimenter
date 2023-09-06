# library imports
import os
import json
import time
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# project imports
from utils.consts import *
from utils.fitness_methods import *
from utils.logger_config import Logger


class Experiments:
    """
    A father class to the experiments, responsible for the run function for all cases.
    Here, only the clustering algorithm changes from case to case.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(clustering_bool: bool,
            scaling_bool: bool, 
            SHAP_bool: bool,
            manifold_embedding_bool: bool,
            dimensionality_reduction_bool: bool,
            algo_str: str,
            results_folder: str,
            data_path: str,
            clustering_algo,
            numerical_run_times: int,
            n_clusters: int
            ):

        # config logging
        start_time = time.time()

        # prepare IO
        os.makedirs(results_folder, exist_ok=True)
        Logger(os.path.join(os.path.dirname(os.path.dirname(__file__)), results_folder, "run.log"))

        # 1) load data
        data = pd.read_csv(data_path)
        Logger.print('Loaded data:\n{}'.format(data))
        #Logger.print('Describe loaded data:\n{}'.format(data.describe()))
        #Logger.print('Nan value check: {}'.format(data.isnull().sum().sum()))
        #Logger.print('Inf value check: {}'.format(np.isinf(data).values.sum()))

        X = data 
        y = np.ones(X.shape[0])

        # 1.1) transform (normalize, standardize, ... ) and split data
        if scaling_bool:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(df_scaled, columns=X.columns)

        train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(X, 
                                                                                y,
                                                                                shuffle=True,
                                                                                test_size=TEST_SIZE_PORTION,
                                                                                random_state=RANDOM_STATE)
        
        Logger.print('Train/Test split data:\n{}'.format(train_data_x))
        Logger.print('Nan check:\n{}'.format(train_data_x.isnull().sum().sum()))
        
        # 1.2) log elapsed time
        data_end_time = time.time()
        Logger.print("Data Loading Finished. Elapsed time: {} ---".format(timedelta(seconds=data_end_time - start_time)))

        # 2) Dimensionality Reduction 
        if dimensionality_reduction_bool:
            Logger.print('Running Dimensionality Reduction:')

            # 2.1) log elapsed time
            dimensionality_reduction_end_time = time.time()
            Logger.print("Dimensionality Reduction Finished. Elapsed time: {}".format(timedelta(seconds=dimensionality_reduction_end_time - data_end_time)))
        
        # 3) Manifold Embedding 
        if manifold_embedding_bool:
            Logger.print('Running Manifold Embedding:')
            
            # 3.1) log elapsed time
            manifold_embedding_end_time = time.time()
            Logger.print("Manifold Embedding Finished. Elapsed time: {}".format(timedelta(seconds=manifold_embedding_end_time - data_end_time)))
        
        # 4) SHAP values
        if SHAP_bool:
            Logger.print('Running SHAP:')
            
            # 4.1) log elapsed time
            SHAP_end_time = time.time()
            Logger.print("SHAP Finished. Elapsed time: {}".format(timedelta(seconds=SHAP_end_time - data_end_time)))

        # 5) Clustering 
        if clustering_bool:
            Logger.print('Running Clustering:')
            results = clustering_algo.run_and_analyze(run_times=numerical_run_times,
                                                     n_clusters=n_clusters,
                                                     train_data_x=X,#train_data_x,
                                                     train_data_y=train_data_y,
                                                     test_data_x=test_data_x,
                                                     test_data_y=test_data_y,
                                                     #generations=numerical_generations,
                                                     #population_size=numerical_population,
                                                     #k_fold=k_fold,
                                                     performance_metric=neg_mean_squared_error_scorer,
                                                     save_dir=results_folder,
                                                     n_jobs=1)

            # 5.1) log elapsed time
            clustering_end_time = time.time()
            Logger.print("Clustering Finished. Elapsed time: {} ---".format(timedelta(seconds=clustering_end_time - data_end_time)))

        # 6) alert results to the user
        Logger.print("\n --- TOTAL ELAPSED TIME: {} ---".format(timedelta(seconds=time.time() - start_time)))
