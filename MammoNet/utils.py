import os

def get_cancer_type_from_path(path):
    return path.split(os.sep)[-3]

def get_resolutions_from_path(path):
    return path.split(os.sep)[-1]