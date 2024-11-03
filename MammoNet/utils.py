import os
import wandb

def get_cancer_type_from_path(path):
    return path.split(os.sep)[-4]

def get_resolutions_from_path(path):
    return path.split(os.sep)[-2]

def setup_wandb(project_name, config_params):
    """
    Set up Weights & Biases (wandb) for logging.

    Args:
        project_name (str): The name of the wandb project.
        config_params (dict): A dictionary of configuration parameters to log.
    """
    # Initialize a new wandb run
    wandb.init(project=project_name)

    # Set configuration parameters
    config = wandb.config
    for key, value in config_params.items():
        config[key] = value

    print("wandb setup complete!")