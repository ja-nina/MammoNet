PATH_TO_DATASET = 'data/BreaKHis_v1/histology_slides/breast'
CLASSES = ['benign', 'malignant']

SUBCLASSES_BENIGN = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
SUBCLASSES_MALIGNANT = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']

ZOOM = ['40', '100', '200', '400']

PATH_TO_DATASET = '../data/BreaKHis_v1/histology_slides/breast'
AUGMENTATION_DIR = '../data/augmented_images'

RESULTS_DIR = '/trained_models'
WANDB_PROJECT = 'MammoNet'
SEED = 2137