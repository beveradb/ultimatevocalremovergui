import json

# Initialize an empty dictionary to hold default parameters for the model.
default_param = {}
# Default values for various parameters, some are specific to training phase.
default_param["bins"] = -1
default_param["unstable_bins"] = -1  # Used only during training
default_param["stable_bins"] = -1  # Used only during training
default_param["sample_rate"] = 44100  # Default sample rate for audio processing.
default_param["pre_filter_start"] = -1
default_param["pre_filter_stop"] = -1
default_param["band"] = {}  # A dictionary to hold band-specific parameters.

# Constant for the number of bins parameter.
N_BINS = "n_bins"


def int_keys(dictionary):
    """
    Converts keys in the given dictionary from strings to integers if they are digit strings.

    Args:
        dictionary (dict): The dictionary whose keys need to be converted.

    Returns:
        dict: A new dictionary with keys converted to integers where applicable.
    """
    result = {}
    for key, value in dictionary.items():
        # Convert key to integer if it's a digit string.
        if key.isdigit():
            key = int(key)
        result[key] = value
    return result


class ModelParameters(object):
    """
    A class to manage model parameters, including loading from a configuration file.

    Attributes:
        param (dict): Dictionary holding all parameters for the model.
    """

    def __init__(self, config_path=""):
        """
        Initializes the ModelParameters object by loading parameters from a JSON configuration file.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        # Load parameters from the given configuration file path.
        with open(config_path, "r") as file:
            self.param = json.loads(file.read(), object_pairs_hook=int_keys)

        # Ensure certain parameters are set to False if not specified in the configuration.
        for key in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_width", "stereo_noise", "reverse"]:
            if key not in self.param:
                self.param[key] = False

        # If 'n_bins' is specified in the parameters, it's used as the value for 'bins'.
        if N_BINS in self.param:
            self.param["bins"] = self.param[N_BINS]
