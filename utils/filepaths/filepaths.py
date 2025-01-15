from pathlib import Path

# Filepaths for case 101
FILE_PATH_CASE_101 = Path("Data", "FieldDataPickled", "case101")
FILE_PATH_CASE_101_HFM_T = Path(
    "Data", "ModelDataPickled", "case101", "model_t_df.pickle"
)
FILE_PATH_CASE_101_HFM_T_OBS = Path(
    "Data", "ModelDataPickled", "case101", "model_t_obs_df.pickle"
)
FILE_PATH_CASE_101_HFM_Z = Path(
    "Data", "ModelDataPickled", "case101", "model_z_df.pickle"
)
FILE_PATH_FLOW_LOOP_DATA = Path(
    "Data", "FlowLoopData", "FlowLoopResults20231012_full.txt"
)
FILE_PATH_DRILLING_REPORT_LABELED_CASE_101 = Path(
    "..", "..", "Master", "LabelingAvBrønnData", "case101", "case101.txt"
)


# Filepaths for case 102
FILE_PATH_CASE_102 = Path("Data", "FieldDataPickled", "case102")

FILE_PATH_CASE_102_HFM_T = Path(
    "Data", "ModelDataPickled", "case102", "model_t_df.pickle"
)
FILE_PATH_CASE_102_HFM_T_OBS = Path(
    "Data", "ModelDataPickled", "case102", "model_t_obs_df.pickle"
)
FILE_PATH_CASE_102_HFM_Z = Path(
    "Data", "ModelDataPickled", "case102", "model_z_df.pickle"
)
FILE_PATH_DRILLING_REPORT_LABELED_CASE_102 = Path(
    "..", "..", "Master", "LabelingAvBrønnData", "case102", "case102.txt"
)

# Filepaths for case 102b
FILE_PATH_CASE_102b = Path("Data", "FieldDataPickled", "case102")

FILE_PATH_CASE_102b_HFM_T = Path(
    "Data", "ModelDataPickled", "case102b", "model_t_df.pickle"
)
FILE_PATH_CASE_102b_HFM_T_OBS = Path(
    "Data", "ModelDataPickled", "case102b", "model_t_obs_df.pickle"
)
FILE_PATH_CASE_102b_HFM_Z = Path(
    "Data", "ModelDataPickled", "case102b", "model_z_df.pickle"
)

# Filepaths for flow loop experiments
FILE_PATH_FLOW_LOOP_DATA = Path(
    "Data", "FlowLoopData", "FlowLoopResults20231012_full.txt"
)

FILE_PATH_BEST_MODELS_FEATURE_1 = Path('Notebooks', 'best_sklearn_model_parameters_feature_1','best_model_params.pkl')
FILE_PATH_BEST_MODELS_REGRESSION_FEATURE_1 = Path('Notebooks', 'best_sklearn_model_parameters_feature_1','best_model_params_regression.pkl')
FILE_PATH_NN_FEATURE_1 = Path("Notebooks","models","nn_on_flow_loop_data_feature_1_no_regula","mlp final classification")
FILE_PATH_NN_REGRESION_FEATURE_1 = Path("Notebooks","models","nn_on_flow_loop_data_feature_1_no_regula","mlp final regression")

FILE_PATH_BEST_MODELS_FEATURE_2 = Path('Notebooks', 'best_sklearn_model_parameters_feature_2','best_model_params.pkl')
FILE_PATH_BEST_MODELS_REGRESSION_FEATURE_2 = Path('Notebooks', 'best_sklearn_model_parameters_feature_2','best_model_params_regression.pkl')
FILE_PATH_NN_FEATURE_2 = Path("Notebooks","models","nn_on_flow_loop_data_feature_2_no_regula","mlp final classification")
FILE_PATH_NN_REGRESION_FEATURE_2 = Path("Notebooks","models","nn_on_flow_loop_data_feature_2_no_regula","mlp final regression")

FILE_PATH_MODEL_RESULTS_FEATURE_1 = Path('Notebooks', 'models', 'model_results_feature_1_no_regula', 'data_classification')
FILE_PATH_MODEL_RESULTS_REGRESSION_FEATURE_1 = Path('Notebooks', 'models', 'model_results_feature_1_no_regula', 'data_regression')

FILE_PATH_MODEL_RESULTS_FEATURE_2 = Path('Notebooks', 'models', 'model_results_feature_2_no_regula', 'data_classification')
FILE_PATH_MODEL_RESULTS_REGRESSION_FEATURE_2 = Path('Notebooks', 'models', 'model_results_feature_2_no_regula', 'data_regression')

FILE_PATH_TEST_SET_CASE_102 = Path("Data","TestDataCase102","test_set_case.pickle")
FILE_PATH_TEST_SET_CASE_101 = Path("Data","TestDataCase101","test_set_case.pickle")


RHEOLOGY_DATA_CASE_101_PATH = Path(
    "Data", "ModelDataPickled", "case101", "RheologyFit.out"
    )
RHEOLOGY_DATA_CASE_102_PATH = Path(
    "Data", "ModelDataPickled", "case102", "RheologyFit.out"
    )


FILE_PATH_TRANSIENT_TRAINING_DATA_102_FEATURES = Path("Data", "Training_transient", "102", "features.pkl")
FILE_PATH_TRANSIENT_TRAINING_DATA_102_TARGETS = Path("Data", "Training_transient", "102", "targets.pkl")
FILE_PATH_TRANSIENT_OTHER_VARIABLES_102 = Path("Data", "Training_transient", "102", "other_variables.pkl")

FILE_PATH_TRANSIENT_TRAINING_DATA_101_FEATURES = Path("Data", "Training_transient", "101", "features.pkl")
FILE_PATH_TRANSIENT_TRAINING_DATA_101_TARGETS = Path("Data", "Training_transient", "101", "targets.pkl")
FILE_PATH_TRANSIENT_OTHER_VARIABLES_101 = Path("Data", "Training_transient", "101", "other_variables.pkl")

FILE_PATH_TRANSIENT_TRAINING_DATA_102b_FEATURES = Path("Data", "Training_transient", "102b", "features.pkl")
FILE_PATH_TRANSIENT_TRAINING_DATA_102b_TARGETS = Path("Data", "Training_transient", "102b", "targets.pkl")
FILE_PATH_TRANSIENT_OTHER_VARIABLES_102b = Path("Data", "Training_transient", "102b", "other_variables.pkl")
