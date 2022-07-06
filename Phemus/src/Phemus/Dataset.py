from .utils import *

class Dataset:
    """The object representation of the metadata for a dataset.

    This class stores the meta data for a dataset. This is used as
    an efficent way to store all the information needed for Aequitas
    to run.

    Attributes:
        sensitive_param_idx_list -- The list of the id of all sensitive parameters that will be tested. 
            The default is an empty list. 
        sensitive_param_name_list -- The list of the name of all sensitive parameters that will be tested.
            The default is an empty list. 
        model_type -- The type of model that will be trained.
        num_params -- The total number of parameters excluding the parameters that will be tested.
        sensitive_param_idx -- The id of the sensitive parameters that is currently being tested.
        sensitive_param_name -- The name of the sensitive parameters that is currently being tested.
        col_to_be_predicted -- The name of the parameters that will be predicted by the model.
        col_to_be_predicted_idx -- The id of the parameters that will be predicted by the model.
        dataset_dir -- The local directory where the dataset is stored as a .csv file.
        column_names -- The list of all parameter names in the dataset.
        input_bounds -- The list of all input bounds for the dataset.
    """
    def __init__(self, num_params, sensitive_param_idx, model_type, sensitive_param_name,
            col_to_be_predicted, dataset_dir, sensitive_param_idx_list = [], sensitive_param_name_list = []):
        """Initialize Dataset class."""
        self.sensitive_param_idx_list = sensitive_param_idx_list
        self.sensitive_param_name_list = sensitive_param_name_list
        if len(self.sensitive_param_idx_list) == 0:
            self.sensitive_param_idx_list.append(sensitive_param_idx)
            self.sensitive_param_name_list.append(sensitive_param_name)
        
        self.model_type = model_type
        self.num_params = num_params
        self.sensitive_param_idx = sensitive_param_idx # Starts at O
        self.sensitive_param_name = sensitive_param_name
        self.col_to_be_predicted = col_to_be_predicted
        self.dataset_dir = dataset_dir

        self.column_names = get_column_names(dataset_dir)
        self.input_bounds = get_input_bounds(dataset_dir)
        self.col_to_be_predicted_idx = get_idx_of_col_to_be_predicted(dataset_dir, col_to_be_predicted)
    
    def update_sensitive_parameter(self, id):
        """Update the sensitive parameter that will be tested

        Update the sensitive parameter that will be tested with an element of the sensitive_param_idx_list.
    
        Args:
            id -- The id of the new sensitive parameter.
    
        Returns:
            This function does not have a return value.

        Raises:
            This function does not raise an exception.
        """
        self.sensitive_param_idx = self.sensitive_param_idx_list[id]
        self.sensitive_param_name = self.sensitive_param_name_list[id]
