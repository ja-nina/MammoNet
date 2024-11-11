from MammoNet.dataset.data_handler import DataHandler
from MammoNet.models.SimpleCNN import SimpleCNN
from MammoNet.models.ViT import VisionTransformer

if __name__ == "__main__":
    # Preparing Data
    data_handler = DataHandler()
    train_loader, val_loader, test_loader = data_handler.get_dataset_loaders()

    # SimpleCNN model training
    print("Starting training for SimpleCNN...")
    simple_cnn = SimpleCNN(num_classes=2)
    simple_cnn.train(train_loader, val_loader, test_loader)

    # VisionTransformer model training
    print("\nStarting training for VisionTransformer...")
    vit_model = VisionTransformer(num_classes=2)
    vit_model.train(train_loader, val_loader, test_loader)
