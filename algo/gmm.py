# library imports
import os
from time import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.mixture import GaussianMixture

# project imports
from utils.logger_config import Logger

def gmm_bic_score(estimator, train_data_x):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(train_data_x)

#def gmm_aic_score(estimator, X):
#    """Callable to pass to GridSearchCV that will use the AIC score."""
#    # Make it negative since GridSearchCV expects a score to maximize
#    return -estimator.aic(X)
    
def SelBest(arr:list, X:int)->list:
    """Returns the set of X configurations with shorter distance."""
    dx=np.argsort(arr)[:X]
    return arr[dx]

class GMM:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    Gaussian Mixture Model (GMM) algorithm.

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
        Run the GMM algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """
        
        XYZ = train_data_x[['Cx','Cy','Cz']]
        X   = train_data_x[['DELTA_Rxx', 'DELTA_Rxy', 'DELTA_Ryy']]

        #algo = GaussianMixture(n_components=n_clusters, n_init=50, init_params="k-means++", random_state=73)
        algo = GaussianMixture(n_components=n_clusters, n_init=2, init_params="k-means++", random_state=73, covariance_type='diag')
        print('GMM clustering algorithm instantiated')

        t0 = time()
        algo.fit(X)
        t1 = time()

        if hasattr(algo, "labels_"):
            labels = algo.labels_.astype(np.int32)
        else:
            labels = algo.predict(X)
 
        means = algo.means_
        covariances = algo.covariances_

        plt.figure(figsize=(30, 15))
        plt.title("GMM", size=18)

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
        cluster_algo_file_path = os.path.join(save_dir, 'GMM.eps')
        plt.savefig(cluster_algo_file_path)
        plt.close()
       
        # compute and tabularize AIC and BIC for number of clusters selection
        param_grid = {
            "n_components": range(1, 10),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
       
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
       
        grid_search.fit(X)
       
        df = pd.DataFrame(grid_search.cv_results_)[
            ["param_n_components", "param_covariance_type", "mean_test_score"]
        ]
       
        df["mean_test_score"] = -df["mean_test_score"]
        df = df.rename(
            columns={
                "param_n_components": "Number of components",
                "param_covariance_type": "Type of covariance",
                "mean_test_score": "BIC score",
            }
        )
        df.sort_values(by="BIC score")
 
        # write BIC table
        bic_table_file_path = os.path.join(save_dir, 'BIC_table.csv')
        df.to_csv(bic_table_file_path)

        # compute and plot AIC and BIC for number of clusters selection
        n_clusters=np.arange(1, 10)
        bics=[]
        bics_err=[]
        iterations=10
        for n in n_clusters:
            tmp_bic=[]
            for _ in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(X) 
                tmp_bic.append(gmm.bic(X))
            val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
            err=np.std(tmp_bic)
            bics.append(val)
            bics_err.append(err)

        plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
        plt.title("BIC Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.legend()    

        # save BIC plot
        bic_plot_file_path = os.path.join(save_dir, 'BIC_plot.eps')
        plt.savefig(bic_plot_file_path)
        plt.close()

        plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
        plt.title("Gradient of BIC Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("grad(BIC)")
        plt.legend()
        
        # save BIC gradient plot
        bic_gradient_plot_file_path = os.path.join(save_dir, 'BIC_gradient_plot.eps')
        plt.savefig(bic_gradient_plot_file_path)
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
        Logger.print("\nFinished GMM clustering run")
        return results

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<GMM>"
