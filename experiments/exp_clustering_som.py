# library imports

# project imports
from utils.consts import *
from experiments.exp_clustering import Experiments
from algo.som import _SOM_


class ExpSOM(Experiments):
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
                        results_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), SOM_RESULTS_FOLDER_NAME),
                        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), SOM_DATA_FOLDER_NAME),
                        #data_generation_function=SOMDataGenerator.generate_kmeans,
                        clustering_algo=_SOM_,
                        numerical_run_times=SOM_NUMERICAL_RUN_TIMES,
                        n_clusters=SOM_N_CLUSTERS
                        #numerical_generations=SOM_NUMERICAL_GENERATION_COUNT,
                        #numerical_population=KEANS_NUMERICAL_POP_SIZE,
                        #analytical_run_times=SOM_ANALYTICAL_RUN_TIMES,
                        #analytical_generations=SOM_ANALYTICAL_GENERATION_COUNT,
                        #analytical_population=SOM_NUMERICAL_POP_SIZE,
                        #parsimony_coefficient=SOM_ANALYTICAL_PARSIMONY_COEFFICIENT,
                        #k_fold=K_FOLD,
                        #samples=SOM_NUMERICAL_NUM_SAMPLES,
                        #rhoa_range=SFF_RHOA_RANGE,
                        #rhop_range=SOM_RHOP_RANGE,
                        #nu_range=SOM_NU_RANGE,
                        #re_range=SOM_RE_RANGE,
                        #genetic_algorithm_feature_selection.pygenetic_algorithm_feature_selection.pyexpected_eq="unknown",
                        #ebs_size_range=SOM_DIMENSIONAL_EBS_SIZE_RANGE_1_2
                        )
