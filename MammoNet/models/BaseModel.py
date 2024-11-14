import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from MammoNet.utils.global_variables import RESULTS_DIR, SEED
from MammoNet.utils.utils import create_results_dir, setup_wandb, set_seed


class BaseModel:
    def __init__(self, model, num_classes, model_name, lr=0.001, epochs=10, suffix = None):
        self.model_name = model_name
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.results_dir = RESULTS_DIR
        self.save_model_file= f"{self.model.name}_{suffix}.pth" if suffix is not None else f"{self.model.name}.pth"

        create_results_dir(self.results_dir)
        set_seed(SEED)

    def train_epoch(self, train_loader):
        self.model.train()
        self.model.to(self.device)
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training Batch"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        self.model.to(self.device)
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation Batch", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(val_loader.dataset)
        return running_loss / len(val_loader), accuracy

    def train(self, train_loader, val_loader, test_loader):
        # init wandb
        setup_wandb("mammonet_project", {"model": self.model_name, "epochs": self.epochs})

        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)
            test_loss, test_accuracy = self.validate(test_loader)
            print(
                f"Epoch [{epoch + 1}/{self.epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.4f}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, self.save_model_file))
        wandb.finish()

    def load_model(self, model_path, device):
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()
