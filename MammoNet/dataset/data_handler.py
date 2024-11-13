import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import DatasetDict
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from MammoNet.dataset.dataset import HistologyDataset
from MammoNet.utils.global_variables import CLASSES, PATH_TO_DATASET, AUGMENTATION_DIR
from MammoNet.utils.utils import (
    get_cancer_type_from_path,
    get_resolutions_from_path,
    get_label_from_augmented_image_path,
)
from MammoNet.dataset.image_augmentations import ImageAugmentations


class DataHandler:
    def __init__(
        self,
        data_path=PATH_TO_DATASET,
        classes=CLASSES,
        augment=True,
        reuse_augmentation=True,
        num_workers=4,
        batch_size=32,
    ):
        """
        Initialize the DataHandler with the dataset path and classes.
        Reuse augmentation only if random seed is not changed.
        """
        self.classes = classes
        self.data_path = data_path

        # Configure augmentation
        self.augment = augment
        self.reuse_augmentation = reuse_augmentation
        self.augmentation_dir = AUGMENTATION_DIR
        self.num_copies = 1
        self.num_workers = num_workers
        self.batch_size = batch_size

        os.makedirs(self.augmentation_dir, exist_ok=True)

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
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.endswith(".png"):
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
        test_labels, _, _ = zip(*test_combined_labels)

        return train_files, val_files, test_files, train_labels, val_labels, test_labels

    def generate_augmented_images(self, input_images_paths, input_labels, num_copies=1):
        """
        Generate augmented images for the input images.
        """
        augmented_images = []
        augmented_labels = []

        if self.augment:
            if not self.reuse_augmentation:
                for idx, (label, input_image_path) in tqdm(
                    enumerate(zip(input_labels, input_images_paths)), total=len(input_labels), desc="Augmenting images"
                ):
                    img = Image.open(input_image_path)
                    img_array = np.array(img)

                    augmenter = ImageAugmentations()

                    for _ in range(num_copies):
                        augmented_img_pil = augmenter(img_array)

                        output_path = os.path.join(self.augmentation_dir, f"augmented_{idx}_{label}.png")

                        augmented_img_pil.save(output_path)

                        augmented_images.append(output_path)
                        augmented_labels.append(label)
            else:
                augmented_images = [
                    os.path.join(self.augmentation_dir, file)
                    for file in os.listdir(self.augmentation_dir)
                    if file.endswith(".png")
                ]
                augmented_labels = [get_label_from_augmented_image_path(file) for file in augmented_images]

        return augmented_images, augmented_labels

    def create_datasets_with_augmentation(
        self, train_files, val_files, test_files, train_labels, val_labels, test_labels
    ):
        """
        Create datasets with augmentation for training, validation, and testing.
        """
        augmented_files, augmented_labels = self.generate_augmented_images(train_files, train_labels)
        train_files = train_files + augmented_files
        train_labels = tuple(list(train_labels) + augmented_labels)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = HistologyDataset(train_files, train_labels, transform=transform)
        val_dataset = HistologyDataset(val_files, val_labels, transform=transform)
        test_dataset = HistologyDataset(test_files, test_labels, transform=transform)

        return DatasetDict({"train": train_dataset, "test": test_dataset, "valid": val_dataset})

    def create_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """
        Create data loaders for training, validation, and testing datasets.
        """
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader, test_loader

    def get_dataset_loaders(self, random_seed=42):
        """
        Get dataset loaders for training, validation, and testing datasets.
        """
        
        file_paths, labels, sublabels, resolutions = self.read_full_dataset()
        train_files, val_files, test_files, train_labels, val_labels, test_labels = self.create_stratified_datasets(
            file_paths, labels, sublabels, resolutions, random_seed=random_seed
        )
        datasets = self.create_datasets_with_augmentation(
            train_files, val_files, test_files, train_labels, val_labels, test_labels
        )
        train_loader, val_loader, test_loader = self.create_data_loaders(
            datasets["train"], datasets["valid"], datasets["test"]
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_handler = DataHandler(augment=True, reuse_augmentation=False)
    train_loader, val_loader, test_loader = data_handler.get_dataset_loaders(augment=False)
