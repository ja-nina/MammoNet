import os
from datasets import DatasetDict
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from MammoNet.dataset.dataset import HistologyDataset
from MammoNet.global_variables import CLASSES, PATH_TO_DATASET
from MammoNet.utils import get_cancer_type_from_path, get_resolutions_from_path
from collections import Counter

class DataHandler:
    def __init__(self, data_path=PATH_TO_DATASET, classes=CLASSES):
        """
        Initialize the DataHandler with the dataset path and classes.
        """
        self.classes = classes
        self.data_path = data_path
        
    def read_full_dataset(self):
        """
        Read the dataset and collect file paths, labels, sublabels, and resolutions.
        """
        file_paths = []
        labels = []
        sublabels = []
        resolutions = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.endswith('.png'):
                        normalized_path = os.path.normpath(os.path.join(root, file))
                        file_paths.append(normalized_path)
                        labels.append(class_name)
                        sublabels.append(get_cancer_type_from_path(normalized_path))
                        resolutions.append(get_resolutions_from_path(normalized_path))

        return file_paths, labels, sublabels, resolutions

    def create_stratified_datasets(self, file_paths, labels, sublabels, resolutions, random_seed=42):
        """
        Create stratified datasets for training, validation, and testing.
        """
        combined_labels = list(zip(labels, sublabels, resolutions))

        train_files, temp_files, train_combined_labels, temp_combined_labels = train_test_split(
            file_paths, combined_labels, test_size=0.4, stratify=combined_labels, random_state=random_seed
        )
        val_files, test_files, val_combined_labels, test_combined_labels = train_test_split(
            temp_files, temp_combined_labels, test_size=0.5, stratify=temp_combined_labels, random_state=random_seed
        )
        
        train_labels, _, _ = zip(*train_combined_labels)
        val_labels, _, _ = zip(*val_combined_labels)
        test_labels,  _, _  = zip(*test_combined_labels)
        
        return train_files, val_files, test_files, train_labels, val_labels, test_labels

    def create_datasets_with_augmentation(self, train_files, val_files, test_files, train_labels, val_labels, test_labels):
        """
        Create datasets with augmentation for training, validation, and testing.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = HistologyDataset(train_files, train_labels, transform=transform)
        val_dataset = HistologyDataset(val_files, val_labels, transform=transform)
        test_dataset = HistologyDataset(test_files, test_labels, transform=transform)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'valid': val_dataset
        })

    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        """
        Create data loaders for training, validation, and testing datasets.
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def get_dataset_loaders(self, random_seed=42):
        """
        Get dataset loaders for training, validation, and testing datasets.
        """
        file_paths, labels, sublabels, resolutions = self.read_full_dataset()
        train_files, val_files, test_files, train_labels, val_labels, test_labels = self.create_stratified_datasets(file_paths, labels, sublabels, resolutions, random_seed=random_seed)    
        datasets = self.create_datasets_with_augmentation(train_files, val_files, test_files, train_labels, val_labels, test_labels)
        train_loader, val_loader, test_loader = self.create_data_loaders(datasets['train'], datasets['valid'], datasets['test'])
        
        return train_loader, val_loader, test_loader
    

if __name__ == '__main__':
    
    # naive test
    data_handler = DataHandler()
    data_handler.get_dataset_loaders()