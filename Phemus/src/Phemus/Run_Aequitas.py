from .Retrain_Sklearn import retrain_sklearn
from .Aequitas_Fully_Directed_Sklearn import aequitas_fully_directed_sklearn
from .Aequitas_Semi_Directed_Sklearn import aequitas_semi_directed_sklearn
from .Aequitas_Random_Sklearn import aequitas_random_sklearn
from .Generate_Sklearn_Classifier import generate_sklearn_classifier

from .Dataset import Dataset

def run_aequitas_once(dataset, perturbation_unit, pkl_dir, improved_pkl_dir, retrain_csv_dir, 
                        plot_dir, mode = "Random", threshold = 0, global_iteration_limit = 1000, 
                        local_iteration_limit = 100, num_trials = 100, samples = 100):
    """Run the complete Aequitas process on a single sensitive parameter

    Perform the three required steps for improving the fairness of a machine 
    learning model from a dataset. run_aequitas_once will first train a model
    from the dataset, and then perform a selected mode of Aequitas to identify
    discrimitory input, and then retrain the model using the discrimitory inputs
    identified.
    
    Args:
        dataset -- A dataset object that contains the meta information about the dataset used.
        perturbation_unit -- A unit measurement for each perturbation step.
        pkl_dir --  The directory where the initially trained model in .pkl formate will be stored.
        improved_pkl_dir -- The directory where the model with improved fairness in .pkl formate will be stored.
        retrain_csv_dir -- The directory where the discrimitory inputs in .csv formate will be stored.
        plot_dir -- The directory where the plot of fairness increase will be stored in.
        mode -- The mode of Aequitas that will be ran. There are three possible modes: Random, Semi, and Fully
            The default mode is set to Random.
        threshold -- A quantitative threshold used to determine whether a input is discrimitory or not. 
            For a binary classification, this threshold should be zero. The default threshold is set to zero.
        global_iteration_limit -- The number of time where global discovery will be ran. The default 
            limit is set to 1000.
        local_iteration_limit -- The number of time where local perturbation will be ran for each 
            input discovered in global discovery. The default limit is set to 100.
        num_trials -- The number of trials that will be conducted when doing fairness estimation. The default
            trial is set to 100. 
        samples -- The number of samples that will be taken when doing fairness estimation. The default
            sample time is set to 100.
    
    Returns:
        This function does not have a return value.

    Raises:
        ValueError: An error occurs when a not supported Aequitas mode is selected.
    """
    if mode == "Random":
        aequitas_random_sklearn(dataset, perturbation_unit, threshold, global_iteration_limit, 
                                    local_iteration_limit, pkl_dir, retrain_csv_dir)
    elif mode == "Semi":
        aequitas_semi_directed_sklearn(dataset, perturbation_unit, threshold, global_iteration_limit, 
                                    local_iteration_limit, pkl_dir, retrain_csv_dir)
    elif mode == "Fully":
        aequitas_fully_directed_sklearn(dataset, perturbation_unit, threshold, global_iteration_limit, 
                                    local_iteration_limit, pkl_dir, retrain_csv_dir)
    else:
        raise ValueError("Mode of Aequitas selected is not valid. Possible modes are Random, Semi and Fully")
    
    retrain_sklearn(dataset, pkl_dir, retrain_csv_dir, improved_pkl_dir, plot_dir, threshold, num_trials, samples)

def run_aequitas(dataset: Dataset, perturbation_unit, pkl_dir, improved_pkl_dir, retrain_csv_dir, plot_dir, \
        mode = "Random", threshold = 0, global_iteration_limit = 1000, local_iteration_limit = 100, num_trials = 100, samples = 100):
    """Run the complete Aequitas process on all sensitive parameters

    Perform the three required steps on all senstive parameters
    for improving the fairness of a machine learning model from a dataset. 
    run_aequitas_once will first train a model from the dataset, 
    and then perform a selected mode of Aequitas to identify
    discrimitory input, and then retrain the model using the discrimitory inputs
    identified.
    
    Args:
        dataset -- A dataset object that contains the meta information about the dataset used.
        perturbation_unit -- A unit measurement for each perturbation step.
        pkl_dir --  The directory where the initially trained model in .pkl formate will be stored.
        improved_pkl_dir -- The directory where the model with improved fairness in .pkl formate will be stored.
        retrain_csv_dir -- The directory where the discrimitory inputs in .csv formate will be stored.
        plot_dir -- The directory where the plot of fairness increase will be stored in.
        mode -- The mode of Aequitas that will be ran. There are three possible modes: Random, Semi, and Fully
            The default mode is set to Random.
        threshold -- A quantitative threshold used to determine whether a input is discrimitory or not. 
            For a binary classification, this threshold should be zero. The default threshold is set to zero.
        global_iteration_limit -- The number of time where global discovery will be ran. The default 
            limit is set to 1000.
        local_iteration_limit -- The number of time where local perturbation will be ran for each 
            input discovered in global discovery. The default limit is set to 100.
        num_trials -- The number of trials that will be conducted when doing fairness estimation. The default
            trial is set to 100. 
        samples -- The number of samples that will be taken when doing fairness estimation. The default
            sample time is set to 100.
    
    Returns:
        This function does not have a return value.

    Raises:
        This function does not rasie an exception.
    """
    num_of_sens_param = len(dataset.sensitive_param_idx_list)

    generate_sklearn_classifier(dataset, pkl_dir)

    for i in range(num_of_sens_param):
        dataset.update_sensitive_parameter(i)
        run_aequitas_once(dataset, perturbation_unit, pkl_dir, improved_pkl_dir, \
            retrain_csv_dir, plot_dir, mode, threshold, \
            global_iteration_limit, local_iteration_limit, num_trials, samples)


