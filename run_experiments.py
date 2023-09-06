# library imports
import os

# project imports
from experiments.exp_clustering_kmeans import ExpKmeans
from experiments.exp_clustering_minibatchkmeans import ExpMiniBatchKMeans
from experiments.exp_clustering_bisectingkmeans import ExpBisectingKMeans
from experiments.exp_clustering_robustweightedkmeans import ExpRobustWeightedKMeans
from experiments.exp_clustering_dbscan import ExpDBSCAN
from experiments.exp_clustering_optics import ExpOPTICS
from experiments.exp_clustering_birch import ExpBIRCH
from experiments.exp_clustering_hdbscan import ExpHDBSCAN
from experiments.exp_clustering_meanshift import ExpMeanShift
from experiments.exp_clustering_affinitypropagation import ExpAffinityPropagation
from experiments.exp_clustering_agglomerative import ExpAgglomerativeClustering
from experiments.exp_clustering_fcm import ExpFCM
from experiments.exp_clustering_som import ExpSOM
from experiments.exp_clustering_spectral import ExpSpectralClustering
from experiments.exp_clustering_kmedoids import ExpKmedoids
from experiments.exp_clustering_gmm import ExpGaussianMixtureModels

class ExperimentsRunner:
    """
    Single entry point for the project.
    This file runs all the experiments in the project 
    and save the raw results for the paper/beamer
    """

    # CONSTS #
    RESULTS_FOLDER_NAME = "results"

    def __init__(self):
        pass

    #@staticmethod
    #def run(scaling_bool: bool = True, # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    #        dimensionality_reduction_bool: bool = False, # https://scikit-learn.org/stable/modules/decomposition.html#factor-analysis
    #        manifold_embedding_bool: bool = False, # https://scikit-learn.org/stable/modules/manifold.html
    #        SHAP_bool: bool = False, # https://shap.readthedocs.io/en/latest/index.html
    #        clustering_bool: bool = True,
    #        algo_str: str = None
    #        ):

    @staticmethod
    def run(clustering_bool: bool = True,
            scaling_bool: bool = True,
            SHAP_bool: bool = True,
            manifold_embedding_bool: bool = True,
            dimensionality_reduction_bool: bool = True,
            algo_str: str = None
            ):
        """
        Single method to use in the class.
        Run the experiments, if requested
        """
        
        # prepare IO
        os.makedirs(os.path.join(os.path.dirname(__file__), ExperimentsRunner.RESULTS_FOLDER_NAME), exist_ok=True)

        # run all the experiments
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="KMEANS":
            ExpKmeans.perform(clustering_bool=clustering_bool,
                              scaling_bool=scaling_bool,
                              SHAP_bool=SHAP_bool,
                              manifold_embedding_bool=manifold_embedding_bool,
                              dimensionality_reduction_bool=dimensionality_reduction_bool,
                              algo_str=algo_str)
       
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="GMM":
            ExpGaussianMixtureModels.perform(clustering_bool=clustering_bool,
                                             scaling_bool=scaling_bool,
                                             SHAP_bool=SHAP_bool,
                                             manifold_embedding_bool=manifold_embedding_bool,
                                             dimensionality_reduction_bool=dimensionality_reduction_bool,
                                             algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="KMEDOIDS":
            ExpKmedoids.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="MINIBATCHKMEANS":
            ExpMiniBatchKMeans.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)

        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="BISECTINGKMEANS":
            ExpBisectingKMeans.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="ROBUSTWEIGHTEDKMEANS":     
            ExpRobustWeightedKMeans.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="SPECTRALCLUSTERING":
            ExpSpectralClustering.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="DBSCAN":
            ExpDBSCAN.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="BIRCH":
            ExpBIRCH.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="OPTICS":
            ExpOPTICS.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="HDBSCAN":
            ExpHDBSCAN.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)

        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="MEANSHIFT":
            ExpMeanShift.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)

        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="AFFINITYPROPAGATION":
            ExpAffinityPropagation.perform(clustering_bool=clustering_bool,
                                scaling_bool=scaling_bool,
                                SHAP_bool=SHAP_bool,
                                manifold_embedding_bool=manifold_embedding_bool,
                                dimensionality_reduction_bool=dimensionality_reduction_bool,
                                algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="FCM":
            ExpFCM.perform(clustering_bool=clustering_bool,
                           scaling_bool=scaling_bool,
                           SHAP_bool=SHAP_bool,
                           manifold_embedding_bool=manifold_embedding_bool,
                           dimensionality_reduction_bool=dimensionality_reduction_bool,
                           algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="AGGLOMERATIVECLUSTERING":
            ExpAgglomerativeClustering.perform(clustering_bool=clustering_bool,
                           scaling_bool=scaling_bool,
                           SHAP_bool=SHAP_bool,
                           manifold_embedding_bool=manifold_embedding_bool,
                           dimensionality_reduction_bool=dimensionality_reduction_bool,
                           algo_str=algo_str)
        
        if (clustering_bool or scaling_bool or scaling_bool or SHAP_bool or manifold_embedding_bool or dimensionality_reduction_bool) and algo_str=="SOM":
            ExpSOM.perform(clustering_bool=clustering_bool,
                           scaling_bool=scaling_bool,
                           SHAP_bool=SHAP_bool,
                           manifold_embedding_bool=manifold_embedding_bool,
                           dimensionality_reduction_bool=dimensionality_reduction_bool,
                           algo_str=algo_str)

if __name__ == '__main__':
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="KMEANS")

    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="GMM")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="KMEDOIDS")

    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="SOM")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="FCM")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="BISECTINGKMEANS")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="MINIBATCHKMEANS")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="SPECTRALCLUSTERING")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="DBSCAN")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="HDBSCAN")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="OPTICS")
    
    ExperimentsRunner.run(clustering_bool=True,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="AFFINITYPROPAGATION")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="BIRCH")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="ROBUSTWEIGHTEDKMEANS")
    
    ExperimentsRunner.run(clustering_bool=True,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="AGGLOMERATIVECLUSTERING")
    
    ExperimentsRunner.run(clustering_bool=False,
                          scaling_bool=False,
                          SHAP_bool=False,
                          manifold_embedding_bool=False,
                          dimensionality_reduction_bool=False,
                          algo_str="MEANSHIFT")
