from MammoNet.dataset.data_handler import DataHandler
from MammoNet.models.SimpleCNN import SimpleCNN
from MammoNet.models.SimpleNN import SimpleNN
from MammoNet.models.SimplerNN import SimplerNN
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

    # Preparing Data with augmentation
    print("\nPreparing data with augmentation...")
    data_handler_with_sample = DataHandler(augment=True, under_over_sample=True)
    train_loader_with_sample, val_loader_with_sample, test_loader_with_sample = data_handler_with_sample.get_dataset_loaders()

    # SimpleCNN model training on data without augmentation
    print("\nStarting training for SimpleCNN (No Augmentation)...")
    simple_cnn_no_aug = SimpleCNN(num_classes=2, suffix='no_aug')
    simple_cnn_no_aug.train(train_loader_no_aug, val_loader_no_aug, test_loader_no_aug)

    # SimpleCNN model training on data with augmentation
    print("\nStarting training for SimpleCNN (With Augmentation)...")
    simple_cnn_with_aug = SimpleCNN(num_classes=2, suffix='with_aug')
    simple_cnn_with_aug.train(train_loader_with_aug, val_loader_with_aug, test_loader_with_aug)

    # SimpleCNN model training on data with augmentation
    print("\nStarting training for SimpleCNN (With Balancing Augmentation)...")
    simple_cnn_with_sample = SimpleCNN(num_classes=2, suffix='with_balancing_aug')
    simple_cnn_with_sample.train(train_loader_with_sample, val_loader_with_sample, test_loader_with_sample)

    # SimpleNN model training on data without augmentation
    print("\nStarting training for SimpleNN (No Augmentation)...")
    simple_cnn_no_aug = SimpleNN(num_classes=2, suffix='no_aug')
    simple_cnn_no_aug.train(train_loader_no_aug, val_loader_no_aug, test_loader_no_aug)

    # SimpleNN model training on data with augmentation
    print("\nStarting training for SimpleNN (With Augmentation)...")
    simple_cnn_with_aug = SimpleNN(num_classes=2, suffix='with_aug')
    simple_cnn_with_aug.train(train_loader_with_aug, val_loader_with_aug, test_loader_with_aug)

    # SimpleNN model training on data with augmentation
    print("\nStarting training for SimpleNN (With Balancing Augmentation)...")
    simple_cnn_with_sample = SimpleNN(num_classes=2, suffix='with_balancing_aug')
    simple_cnn_with_sample.train(train_loader_with_sample, val_loader_with_sample, test_loader_with_sample)

    # SimplerNN model training on data without augmentation
    print("\nStarting training for SimplerNN (No Augmentation)...")
    simpler_nn_no_aug = SimplerNN(num_classes=2, suffix='no_aug')
    simpler_nn_no_aug.train(train_loader_no_aug, val_loader_no_aug, test_loader_no_aug)

    # SimplerNN model training on data with augmentation
    print("\nStarting training for SimplerNN (With Augmentation)...")
    simpler_nn_with_aug = SimplerNN(num_classes=2, suffix='with_aug')
    simpler_nn_with_aug.train(train_loader_with_aug, val_loader_with_aug, test_loader_with_aug)

    # VisionTransformer model training on data without augmentation
    print("\nStarting training for VisionTransformer (No Augmentation)...")
    vit_model_no_aug = VisionTransformer(num_classes=2, suffix='no_aug')
    vit_model_no_aug.train(train_loader_no_aug, val_loader_no_aug, test_loader_no_aug)

    # VisionTransformer model training on data with augmentation
    print("\nStarting training for VisionTransformer (With Augmentation)...")
    vit_model_with_aug = VisionTransformer(num_classes=2, suffix='with_aug')
    vit_model_with_aug.train(train_loader_with_aug, val_loader_with_aug, test_loader_with_aug)

    # VisionTransformer model training on data with augmentation
    print("\nStarting training for VisionTransformer (With Balancing Augmentation)...")
    vit_model_with_aug = VisionTransformer(num_classes=2, suffix='with_balancing_aug')
    vit_model_with_aug.train(train_loader_with_sample, val_loader_with_sample, test_loader_with_sample)
