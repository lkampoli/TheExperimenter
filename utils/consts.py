# library imports
import os

# 1) General for all experiment:
DATA_FOLDER = "data"
RESULTS_FOLDER = "results"
g_force = 9.81
REL_ERR_OF_STD = 0.05
DEFAULT_FIG_SIZE = 8
DEFAULT_DPI = 600
FEATURE_IMPORTANCE_SIMULATION_COUNT = 5 #100
JSON_INDENT = 2
K_FOLD = 5
RANDOM_STATE = 73
SYMBOLIC_PERCENT_OF_MAJORITY = 0.6
SYMBOLIC_P_VALUE_THRESHOLD = 0.8
SYMBOLIC_EQ_RANKING_METRIC = "r2"
SYMBOLIC_TOP_EQS_MAX_NUM = 5

# KMEANS
KMEANS_DROP_PARAM = ""
KMEANS_NUMERICAL_RUN_TIMES = 1
KMEANS_N_CLUSTERS = 3
KMEANS_N_SAMPLES_STR = str(37000)
KMEANS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "means_results")
#KMEANS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "kmeans_results", "{}_samples".format(KMEANS_N_SAMPLES_STR))
KMEANS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "kmeans_data_" + "{}_samples.csv".format(KMEANS_N_SAMPLES_STR))
KMEANS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# MINIBATCHKMEANS
MINIBATCHKMEANS_DROP_PARAM = ""
MINIBATCHKMEANS_NUMERICAL_RUN_TIMES = 1
MINIBATCHKMEANS_N_CLUSTERS = 3
MINIBATCHKMEANS_N_SAMPLES_STR = str(37000)
MINIBATCHKMEANS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "minibatchkmeans_results")
#MINIBATCHKMEANS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "minibatchkmeans_results", "{}_samples".format(MINIBATCHKMEANS_N_SAMPLES_STR))
MINIBATCHKMEANS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# BISECTINGKMEANS
BISECTINGKMEANS_DROP_PARAM = ""
BISECTINGKMEANS_NUMERICAL_RUN_TIMES = 1
BISECTINGKMEANS_N_CLUSTERS = 3
BISECTINGKMEANS_N_SAMPLES_STR = str(37000)
BISECTINGKMEANS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "bisectingkmeans_results")
BISECTINGKMEANS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# ROBUSTWEIGHTEDKMEANS
ROBUSTWEIGHTEDKMEANS_DROP_PARAM = ""
ROBUSTWEIGHTEDKMEANS_NUMERICAL_RUN_TIMES = 1
ROBUSTWEIGHTEDKMEANS_N_CLUSTERS = 3
ROBUSTWEIGHTEDKMEANS_N_SAMPLES_STR = str(37000)
ROBUSTWEIGHTEDKMEANS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "robustweightedkmeans_results")
ROBUSTWEIGHTEDKMEANS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# KMEDOIDS
KMEDOIDS_DROP_PARAM = ""
KMEDOIDS_NUMERICAL_RUN_TIMES = 1
KMEDOIDS_N_CLUSTERS = 3
KMEDOIDS_N_SAMPLES_STR = str(37000)
#KMEDOIDS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "kmedoids_results", "{}_samples".format(KMEDOIDS_N_SAMPLES_STR))
KMEDOIDS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "kmedoids_results")
KMEDOIDS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "kmedoids_data_" + "{}_samples.csv".format(KMEDOIDS_N_SAMPLES_STR))
KMEDOIDS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# GMM
GMM_DROP_PARAM = ""
GMM_NUMERICAL_RUN_TIMES = 1
GMM_N_CLUSTERS = 3
GMM_N_SAMPLES_STR = str(37000)
#GMM_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "gmm_results", "{}_samples".format(GMM_N_SAMPLES_STR))
GMM_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "gmm_results")
GMM_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "gmm_data_" + "{}_samples.csv".format(GMM_N_SAMPLES_STR))
GMM_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# SPECTRAL
SPECTRAL_DROP_PARAM = ""
SPECTRAL_NUMERICAL_RUN_TIMES = 1
SPECTRAL_N_CLUSTERS = 3
SPECTRAL_N_SAMPLES_STR = str(37000)
SPECTRAL_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "spectral_results")
SPECTRAL_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# DBSCAN
DBSCAN_DROP_PARAM = ""
DBSCAN_NUMERICAL_RUN_TIMES = 1
DBSCAN_N_CLUSTERS = 3
DBSCAN_N_SAMPLES_STR = str(37000)
DBSCAN_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "dbscan_results")
DBSCAN_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# OPTICS
OPTICS_DROP_PARAM = ""
OPTICS_NUMERICAL_RUN_TIMES = 1
OPTICS_N_CLUSTERS = 3
OPTICS_N_SAMPLES_STR = str(37000)
OPTICS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "optics_results")
OPTICS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# BIRCH
BIRCH_DROP_PARAM = ""
BIRCH_NUMERICAL_RUN_TIMES = 1
BIRCH_N_CLUSTERS = 3
BIRCH_N_SAMPLES_STR = str(37000)
BIRCH_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "birch_results")
BIRCH_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# HDBSCAN
HDBSCAN_DROP_PARAM = ""
HDBSCAN_NUMERICAL_RUN_TIMES = 1
HDBSCAN_N_CLUSTERS = 3
HDBSCAN_N_SAMPLES_STR = str(37000)
HDBSCAN_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "hdbscan_results")
HDBSCAN_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# MEANSHIFN
MEANSHIFT_DROP_PARAM = ""
MEANSHIFT_NUMERICAL_RUN_TIMES = 1
MEANSHIFT_N_CLUSTERS = 3
MEANSHIFT_N_SAMPLES_STR = str(37000)
MEANSHIFT_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "meanshift_results")
MEANSHIFT_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# AFFINITYPROPAGATION
AFFINITYPROPAGATION_DROP_PARAM = ""
AFFINITYPROPAGATION_NUMERICAL_RUN_TIMES = 1
AFFINITYPROPAGATION_N_CLUSTERS = 3
AFFINITYPROPAGATION_N_SAMPLES_STR = str(37000)
AFFINITYPROPAGATION_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "affinitypropagation_results")
AFFINITYPROPAGATION_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# FCM
FCM_DROP_PARAM = ""
FCM_NUMERICAL_RUN_TIMES = 1
FCM_N_CLUSTERS = 3
FCM_N_SAMPLES_STR = str(37000)
FCM_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "fcm_results")
FCM_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# AGGLOMERATIVE
AGGLOMERATIVE_DROP_PARAM = ""
AGGLOMERATIVE_NUMERICAL_RUN_TIMES = 1
AGGLOMERATIVE_N_CLUSTERS = 3
AGGLOMERATIVE_N_SAMPLES_STR = str(37000)
AGGLOMERATIVE_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "agglomerative_results")
AGGLOMERATIVE_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")

# SOM
SOM_DROP_PARAM = ""
SOM_NUMERICAL_RUN_TIMES = 1
SOM_N_CLUSTERS = 3
SOM_N_SAMPLES_STR = str(37000)
SOM_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER, "som_results")
SOM_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, "database.csv")




TEST_SIZE_PORTION = 0.2

## 2) Constant acceleration exp:
## - data generation:
#CONST_ACCELERATION_NUM_SAMPLES = 400
#CONST_ACCELERATION_TEST_SIZE_PORTION = 0.75
## - experiment run:
#CONST_ACCELERATION_NUMERICAL_RUN_TIMES = 20
#CONST_ACCELERATION_NUMERICAL_GENERATION_COUNT = 5
#CONST_ACCELERATION_NUMERICAL_POP_SIZE = 30
#CONST_ACCELERATION_ANALYTICAL_RUN_TIMES = 20
#CONST_ACCELERATION_ANALYTICAL_GENERATION_COUNT = 5
#CONST_ACCELERATION_ANALYTICAL_POP_SIZE = 50
#CONST_ACCELERATION_NOISE_RANGE = (0, 0.02)
#CONST_ACCELERATION_ANALYTICAL_PARSIMONY_COEFFICIENT = 0.02
#CONST_ACCELERATION_EBS_SIZE_RANGE = (5,)
## - result path
#CONST_ACCELERATION_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
#                                                      "constant_acceleration_results",
#                                                      "{}_samples".format(CONST_ACCELERATION_NUM_SAMPLES))
#
## 3) Steady free fall with drag exp:
## - data generation:
#N_FREQ_SUFFIX = ['_11', '_12', '_13', '_21', '_23', '_31', '_32']
#STEADY_FALL_MINIZE_TOL = 1e-25
#SFF_RHOA_RANGE = (998., 1300.)   # fresh water -> salt water at 20C [kg/m3]
#SFF_RHOP_RANGE = (0, 5000)
#SFF_NU_RANGE = (1e-6, 1.4e-6)    # viscosity corresponding to rhoa [m2/s]
#SFF_RE_RANGE = (1., 100.)        # Reynolds range where Cd changes significantly
#SFF_CASE_2_NOISE_RANGE = (0, 0.02)
#SFF_TEST_SIZE_PORTION = 0.2
#SFF_DIMENSIONAL_EBS_SIZE_RANGE_1_2 = (11,)
#SFF_DIMENSIONAL_EBS_SIZE_RANGE_2_easy = (7,)
#SFF_DIMENSIONAL_EBS_SIZE_RANGE_3 = (3,)
#SFF_1_DROP_PARAM = "V"
#FORCE_DATA_OVERRIDE_FLAG = False
## - experiment run:
#SFF_NUMERICAL_NUM_SAMPLES = 10**4
#SFF_NUMERICAL_RUN_TIMES = 20
#SFF_NUMERICAL_GENERATION_COUNT = 3
#SFF_NUMERICAL_POP_SIZE = 25
#SFF_ANALYTICAL_RUN_TIMES = 20
#SFF_ANALYTICAL_GENERATION_COUNT = 10
#SFF_ANALYTICAL_POP_SIZE = 2000
#SFF_ANALYTICAL_PARSIMONY_COEFFICIENT = 0.025
## - feature selection:
#FEATURE_SELECTION_GENERATIONS_COUNT = 2
#FEATURE_SELECTION_POP_SIZE = 8
#FEATURE_SELECTION_MUTATION_RATE = 0.1
#FEATURE_SELECTION_ROYALTY = 0.05
## - data and result paths
#SFF_N_SAMPLES_STR = str(round(SFF_NUMERICAL_NUM_SAMPLES/1000)) + "k"
#SFF_1_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
#                                         "steady_fall_case_1_results",
#                                         "{}_samples_without_{}".format(SFF_N_SAMPLES_STR, SFF_1_DROP_PARAM))
#
#SFF_1_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
#                                      "case_1_steady_fall_with_drag_data_" +
#                                      "{}_samples_no_{}.csv".format(SFF_N_SAMPLES_STR, SFF_1_DROP_PARAM))
#
#SFF_2_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
#                                         "steady_fall_case_2_results_{}_samples".format(SFF_N_SAMPLES_STR))
#
#SFF_2_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
#                                      "case_2_steady_fall_with_drag_data_{}_samples.csv".format(SFF_N_SAMPLES_STR))
#
#SFF_2_WITH_GUESS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
#                                                    "steady_fall_case_2_with_guess_results_{}_samples".format(SFF_N_SAMPLES_STR))
#SFF_2_WITH_GUESS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
#                                                 "case_2_with_guess_steady_fall_with_drag_data_{}_samples.csv".format(SFF_N_SAMPLES_STR))
#
#SFF_3_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
#                                         "steady_fall_case_3_results_{}_samples".format(SFF_N_SAMPLES_STR))
#SFF_3_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
#                                      "case_3_steady_fall_with_drag_data_{}_samples.csv".format(SFF_N_SAMPLES_STR))
#
## 3) Drag force exp:
## - data generation:
#DRAG_FORCE_NUM_SAMPLES = 10000
#DRAG_FORCE_TEST_SIZE_PORTION = 0.75
## - experiment run:
#DRAG_FORCE_NUMERICAL_RUN_TIMES = 20
#DRAG_FORCE_NUMERICAL_GENERATION_COUNT = 5
#DRAG_FORCE_NUMERICAL_POP_SIZE = 30
#DRAG_FORCE_FEATURE_GENERATIONS_COUNT = 5
#DRAG_FORCE_FEATURE_POP_SIZE = 30
#DRAG_FORCE_MUTATION_RATE = 0.1
#DRAG_FORCE_ROYALTY = 0.05
#DRAG_FORCE_ANALYTICAL_RUN_TIMES = 20
#DRAG_FORCE_ANALYTICAL_GENERATION_COUNT = 5
#DRAG_FORCE_ANALYTICAL_POP_SIZE = 50
#DRAG_FORCE_NOISE_RANGE = (0, 0.02)
#DRAG_FORCE_ANALYTICAL_PARSIMONY_COEFFICIENT = 0.02
#DRAG_FORCE_EBS_SIZE_RANGE = (13,)
## - result path
#DRAG_FORCE_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
#                                              "drag_force_results",
#                                              "{}_samples".format(DRAG_FORCE_NUM_SAMPLES))
#
## end - consts #
