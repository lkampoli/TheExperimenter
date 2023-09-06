# library imports

# project imports
from utils.consts import *
from experiments.exp_clustering import Experiments
from algo.robustweightedkmeans import ROBUSTWEIGHTEDKMEANS


class ExpRobustWeightedKMeans(Experiments):
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
                        results_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), ROBUSTWEIGHTEDKMEANS_RESULTS_FOLDER_NAME),
                        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ROBUSTWEIGHTEDKMEANS_DATA_FOLDER_NAME),
                        #data_generation_function=ROBUSTWEIGHTEDKMEANSDataGenerator.generate_kmeans,
                        clustering_algo=ROBUSTWEIGHTEDKMEANS,
                        numerical_run_times=ROBUSTWEIGHTEDKMEANS_NUMERICAL_RUN_TIMES,
                        n_clusters=ROBUSTWEIGHTEDKMEANS_N_CLUSTERS
                        #numerical_generations=ROBUSTWEIGHTEDKMEANS_NUMERICAL_GENERATION_COUNT,
                        #numerical_population=KEANS_NUMERICAL_POP_SIZE,
                        #analytical_run_times=ROBUSTWEIGHTEDKMEANS_ANALYTICAL_RUN_TIMES,
                        #analytical_generations=ROBUSTWEIGHTEDKMEANS_ANALYTICAL_GENERATION_COUNT,
                        #analytical_population=ROBUSTWEIGHTEDKMEANS_NUMERICAL_POP_SIZE,
                        #parsimony_coefficient=ROBUSTWEIGHTEDKMEANS_ANALYTICAL_PARSIMONY_COEFFICIENT,
                        #k_fold=K_FOLD,
                        #samples=ROBUSTWEIGHTEDKMEANS_NUMERICAL_NUM_SAMPLES,
                        #rhoa_range=SFF_RHOA_RANGE,
                        #rhop_range=ROBUSTWEIGHTEDKMEANS_RHOP_RANGE,
                        #nu_range=ROBUSTWEIGHTEDKMEANS_NU_RANGE,
                        #re_range=ROBUSTWEIGHTEDKMEANS_RE_RANGE,
                        #genetic_algorithm_feature_selection.pygenetic_algorithm_feature_selection.pyexpected_eq="unknown",
                        #ebs_size_range=ROBUSTWEIGHTEDKMEANS_DIMENSIONAL_EBS_SIZE_RANGE_1_2
                        )
