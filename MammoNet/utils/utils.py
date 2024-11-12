import os
import random
import numpy as np
import torch
import wandb
from MammoNet.utils.global_variables import WANDB_PROJECT


def get_cancer_type_from_path(path):
    return path.split(os.sep)[-4]


def get_resolutions_from_path(path):
    return path.split(os.sep)[-2]


def create_results_dir(results_dir):
    """
    Create a directory to save results.

    Args:
        results_dir (str): The path to the directory to save results.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


def setup_wandb(project_name, config_params):
    """
    Set up Weights & Biases (wandb) for logging.

    Args:
        project_name (str): The name of the wandb project.
        config_params (dict): A dictionary of configuration parameters to log.
    """
    # Initialize a new wandb run
    wandb.init(project=WANDB_PROJECT)

    # Set configuration parameters
    config = wandb.config
    for key, value in config_params.items():
        config[key] = value

    print("wandb setup complete!")


def get_label_from_augmented_image_path(path):
    return path.split("_")[-1].replace(".png", "")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
