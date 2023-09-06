# library imports

# project imports
from utils.consts import *
from experiments.exp_clustering import Experiments
from algo.agglomerativeclustering import AGGLOMERATIVECLUSTERING


class ExpAgglomerativeClustering(Experiments):
    """
    The first case of the experiments -
    Program receives a dataset that is missing an essential feature
    (particle velocity), that is needed to deduce the target (drag coefficient).

    Failure of both numerical and analytical parts of the program prove
    that if the user neglects to measure a key physical component in the
    unknown physical phenomena, the user is alerted by bad results.
    """

    def __init__(self):
        Experiments.__init__(self)

    @staticmethod
    def perform(clustering_bool: bool,
                scaling_bool: bool,
                SHAP_bool: bool,
                manifold_embedding_bool: bool,
                dimensionality_reduction_bool: bool,
                algo_str: str):
        """
        Entry point
        """
        Experiments.run(clustering_bool=clustering_bool,
                        scaling_bool=scaling_bool,
                        SHAP_bool=SHAP_bool,
                        manifold_embedding_bool=manifold_embedding_bool,
                        dimensionality_reduction_bool=dimensionality_reduction_bool,
                        algo_str=algo_str,
                        results_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), AGGLOMERATIVE_RESULTS_FOLDER_NAME),
                        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), AGGLOMERATIVE_DATA_FOLDER_NAME),
                        #data_generation_function=AGGLOMERATIVECLUSTERINGDataGenerator.generate_kmeans,
                        clustering_algo=AGGLOMERATIVECLUSTERING,
                        numerical_run_times=AGGLOMERATIVE_NUMERICAL_RUN_TIMES,
                        n_clusters=AGGLOMERATIVE_N_CLUSTERS
                        #numerical_generations=AGGLOMERATIVECLUSTERING_NUMERICAL_GENERATION_COUNT,
                        #numerical_population=KEANS_NUMERICAL_POP_SIZE,
                        #analytical_run_times=AGGLOMERATIVECLUSTERING_ANALYTICAL_RUN_TIMES,
                        #analytical_generations=AGGLOMERATIVECLUSTERING_ANALYTICAL_GENERATION_COUNT,
                        #analytical_population=AGGLOMERATIVECLUSTERING_NUMERICAL_POP_SIZE,
                        #parsimony_coefficient=AGGLOMERATIVECLUSTERING_ANALYTICAL_PARSIMONY_COEFFICIENT,
                        #k_fold=K_FOLD,
                        #samples=AGGLOMERATIVECLUSTERING_NUMERICAL_NUM_SAMPLES,
                        #rhoa_range=SFF_RHOA_RANGE,
                        #rhop_range=AGGLOMERATIVECLUSTERING_RHOP_RANGE,
                        #nu_range=AGGLOMERATIVECLUSTERING_NU_RANGE,
                        #re_range=AGGLOMERATIVECLUSTERING_RE_RANGE,
                        #genetic_algorithm_feature_selection.pygenetic_algorithm_feature_selection.pyexpected_eq="unknown",
                        #ebs_size_range=AGGLOMERATIVECLUSTERING_DIMENSIONAL_EBS_SIZE_RANGE_1_2
                        )
