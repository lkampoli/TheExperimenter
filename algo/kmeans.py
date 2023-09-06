# library imports
import os
from time import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

from scipy.cluster.vq import kmeans as sp_kmeans
from scipy.spatial.distance import cdist,pdist
from matplotlib import cm
import matplotlib.cm as cm
cmap = cm.get_cmap("Spectral")

from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance

# project imports
from utils.logger_config import Logger


class KMEANS:
    """
    This class is responsible for generating the clustering
    partitions for a given set of input features, using a 
    K-means algorithm.

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
        Run the KMeans algorithm with some hyper-parameters
        for multiple times and analyze the stability of the results.
        Returns a pandas dataframe of all results and the best model from
        all runs.
        """

        XYZ = train_data_x[['Cx','Cy','Cz']]
        X   = train_data_x[['DELTA_Rxx', 'DELTA_Rxy', 'DELTA_Ryy']]
 
        print(XYZ)
        print(X)

        # compute WCSS score #
        ######################
        wcss = []
        for i in range(1,21):
            kmeans = KMeans(n_clusters=i, n_init=10, init='k-means++', random_state=73)
            kmeans.fit(X)
            labels = kmeans.fit_predict(X)
            #print(labels)
            wcss.append(kmeans.inertia_)
            
        # compute Percentage of variance explained # 
        ############################################
        K_MAX = 20
        KK = range(1,K_MAX+1)
        
        KM = [sp_kmeans(X,k) for k in KK]
        centroids = [cent for (cent,var) in KM]
        D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
        cIdx = [np.argmin(D,axis=1) for D in D_k]
        dist = [np.min(D,axis=1) for D in D_k]
        
        tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
        totss = sum(pdist(X)**2)/X.shape[0]       # The total sum of squares
        betweenss = totss - tot_withinss          # The between-cluster sum of squares
        
        # plots
        kIdx = 2 # hard-coded
        clr = cmap(np.linspace(0,1,10)).tolist()
        mrk = 'os^p<dvh8>+x.'

        fig, ax1 = plt.subplots(figsize=(30,15)) 
        ax1.set_xlabel('Number of clusters') 
        ax1.set_ylabel('Percentage of variance explained (%)', color = 'black') 
        plot_1 = ax1.plot(KK, betweenss/totss*100, 'b*-', color = 'black', linewidth=5, markersize=20) 
        ax1.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=30, markeredgewidth=5, markeredgecolor='r', markerfacecolor='None')
        ax1.tick_params(axis ='y', labelcolor = 'black') 
        ax1.grid(True)
        ax2 = ax1.twinx() 
        ax2.set_ylabel('WCSS', color = 'green') 
        plot_2 = ax2.plot(range(1,21), wcss, color = 'green', linewidth=5) 
        ax2.tick_params(axis ='y', labelcolor = 'green') 
        plt.xticks(np.arange(1, 21, 1.0))
        # save fig
        wcss_score_file_path = os.path.join(save_dir, 'wcss_score.eps')
        plt.savefig(wcss_score_file_path)
        plt.close()

        # compute silhouette score #
        ############################
        silhouette_scores = [] 

        for n_cluster in range(2,20):
            silhouette_scores.append(
                    silhouette_score(X, KMeans(n_clusters = n_cluster).fit_predict(X))) 
    
        # Plotting a bar graph to compare the results 
        k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 
        plt.figure(figsize=(30,15))
        plt.bar(k, silhouette_scores) 
        plt.xlabel('Number of clusters') 
        plt.ylabel('Silhouette Score') 
        # save fig
        silhouette_score_file_path = os.path.join(save_dir, 'silhouette_scores.eps')
        plt.savefig(silhouette_score_file_path)
        plt.close()

        # compute silhouette plots #
        ############################

        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 10]
        for n_clusters in range_n_clusters:
    
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
    
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 
            ax1.set_xlim([-0.2, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
            # Initialize the clusterer with n_clusters value and a random generator seed for reproducibility.
            #kmeans = KMeans(n_clusters=n_clusters, random_state=rng)# lloyd", "elkan", "auto", "full
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=73, verbose=0).fit(X)
            cluster_labels = kmeans.predict(X)
            
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
    
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
                ith_cluster_silhouette_values.sort()
    
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
    
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
    
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
    
                ax1.set_title("Silhouette plot for the clusters.")
                ax1.set_xlabel("Silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
            
                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            
                ax1.set_yticks([]) # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            
                # 2nd Plot showing the actual clusters formed
                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                ax2.scatter(
                    XYZ['Cx'], XYZ['Cy'], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
                )
            
                # Labeling the clusters
                centers = kmeans.cluster_centers_
                
                # Draw white circles at cluster centers
                #ax2.scatter(
                #    centers[:, 0],
                #    centers[:, 1],
                #    marker="o",
                #    c="white",
                #    alpha=1,
                #    s=200,
                #    edgecolor="k",
                #)
            
                #for i, c in enumerate(centers):
                #    ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
            
                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")
            
                plt.suptitle(
                    "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                    % n_clusters,
                    fontsize=14,
                    fontweight="bold",
                )
                plt.tight_layout()
                silhouette_plots_file_path = os.path.join(save_dir, 'silhouette_plots.eps')
                plt.savefig(silhouette_plots_file_path)
            plt.close()

        # Yellowbrick plots
        # https://www.scikit-yb.org/en/latest/api/cluster/index.html
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=73)
        
        visualizer1 = KElbowVisualizer(model, k=(2,21), metric='distortion')
        visualizer2 = KElbowVisualizer(model, k=(2,21), metric='silhouette')
        visualizer3 = KElbowVisualizer(model, k=(2,21), metric='calinski_harabasz')
        visualizer4 = InterclusterDistance(model)

        visualizer1.fit(X)
        visualizer2.fit(X)
        visualizer3.fit(X)
        visualizer4.fit(X)

        yellowbrick_distortion_file_path = os.path.join(save_dir, 'yellowbrick_distortion.eps')
        visualizer1.show(outpath=yellowbrick_distortion_file_path)
        plt.close()

        yellowbrick_silhouette_file_path = os.path.join(save_dir, 'yellowbrick_silhouette.eps')
        visualizer2.show(outpath=yellowbrick_silhouette_file_path)
        plt.close()
        
        yellowbrick_calinski_harabasz_file_path = os.path.join(save_dir, 'yellowbrick_calinski_harabasz.eps')
        visualizer3.show(outpath=yellowbrick_calinski_harabasz_file_path)
        plt.close()
        
        yellowbrick_InterclusterDistance_file_path = os.path.join(save_dir, 'yellowbrick_InterclusterDistance.eps')
        visualizer4.show(outpath=yellowbrick_InterclusterDistance_file_path)
        plt.close()

        algo = KMeans(n_clusters=n_clusters, n_init=100, random_state=73)
        print('clustering algorithm instantiated')

        t0 = time()
        algo.fit(X)
        t1 = time()
        print('clustering data fitted')

        if hasattr(algo, "labels_"):
            labels = algo.labels_.astype(np.int32)
        else:
            labels = algo.predict(X)
        print('labels computed')

        centroids = algo.cluster_centers_
        print('centroids computed')

        plt.figure(figsize=(30, 15))
        plt.title("K-means", size=18)

        #plt.savefig('clustering_algorithms_'+str(point_percentage)+'%_'+str(clusters)+'_clusters_'+'scaled_'+str(scale)+'.pdf')    
    
        #names = list(data.columns.values.tolist())
        #np.savetxt('labels_'+str(names)+'.txt', y_pred, fmt='%i')

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
        # save fig
        cluster_algo_file_path = os.path.join(save_dir, 'KMEANS.eps')
        plt.savefig(cluster_algo_file_path)

        # prepare DF for results
        results = pd.DataFrame()

        for test in range(run_times):
            Logger.print("Clustering run {}/{}".format(test + 1, run_times))

            try:
                results.at[test, "labels"] = labels
            except:
                print("No labels available")

            try:
                results.at[test, "centroids"] = centroids
            except:
                print("No centroids available")

            # update best mae score and model
            #if performance_score < current_best_performance_score:
            #    best_model = model
            #    current_best_performance_score = performance_score

        # Logger.print and save scoring results of all runs
        Logger.print("\nFinished KMEANS clustering run")
        #Logger.print("{}{}".format(results[labels], results[centroids]))
        return results

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<KMEANS>"
