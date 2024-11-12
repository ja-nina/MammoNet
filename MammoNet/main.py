from MammoNet.dataset.data_handler import DataHandler
from MammoNet.models.SimpleCNN import SimpleCNN
from MammoNet.models.ViT import VisionTransformer

if __name__ == "__main__":
    # Preparing Data without augmentation
    print("Preparing data without augmentation...")
    data_handler_no_aug = DataHandler(augment=False)
    train_loader_no_aug, val_loader_no_aug, test_loader_no_aug = data_handler_no_aug.get_dataset_loaders()

    # Preparing Data with augmentation
    print("\nPreparing data with augmentation...")
    data_handler_with_aug = DataHandler(augment=True)
    train_loader_with_aug, val_loader_with_aug, test_loader_with_aug = data_handler_with_aug.get_dataset_loaders()

    # SimpleCNN model training on data without augmentation
    print("\nStarting training for SimpleCNN (No Augmentation)...")
    simple_cnn_no_aug = SimpleCNN(num_classes=2)
    simple_cnn_no_aug.train(train_loader_no_aug, val_loader_no_aug, test_loader_no_aug)

    # SimpleCNN model training on data with augmentation
    print("\nStarting training for SimpleCNN (With Augmentation)...")
    simple_cnn_with_aug = SimpleCNN(num_classes=2)
    simple_cnn_with_aug.train(train_loader_with_aug, val_loader_with_aug, test_loader_with_aug)

    # VisionTransformer model training on data without augmentation
    print("\nStarting training for VisionTransformer (No Augmentation)...")
    vit_model_no_aug = VisionTransformer(num_classes=2)
    vit_model_no_aug.train(train_loader_no_aug, val_loader_no_aug, test_loader_no_aug)

    # VisionTransformer model training on data with augmentation
    print("\nStarting training for VisionTransformer (With Augmentation)...")
    vit_model_with_aug = VisionTransformer(num_classes=2)
    vit_model_with_aug.train(train_loader_with_aug, val_loader_with_aug, test_loader_with_aug)
