import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from MammoNet.global_variables import RESULTS_DIR, SEED
from MammoNet.dataset.data_handler import DataHandler  # for tests
from MammoNet.utils import create_results_dir, setup_wandb, set_seed


class SimplerNNModel(nn.Module):
    def __init__(self, num_classes, input_size=224):
        super(SimplerNNModel, self).__init__()
        self.name = "SimplerNNModel"
        
        flattened_dim = 3 * input_size * input_size

        self.fc1 = nn.Linear(flattened_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimplerNN:
    def __init__(self, num_classes=2, input_size=224):
        self.model = SimplerNNModel(num_classes, input_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.results_dir = RESULTS_DIR
        self.epochs = 10
        
        create_results_dir(self.results_dir)
        setup_wandb("mammonet_project", {"model": "SimplerNN", "epochs": self.epochs})
        set_seed(SEED)

    def train_epoch(self, train_loader):
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
        
    def train(self, train_loader, val_loader, test_loader, random_seed=42):
        
        best_val_loss = torch.inf
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)
            test_loss, test_accuracy  = self.validate(test_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            })
            
            # Apply early stopping criterion
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                           os.path.join(self.results_dir,  f"{self.model.name}.pth"))
                

if __name__ == '__main__':
    data_handler = DataHandler()
    train_loader, val_loader, test_loader = data_handler.get_dataset_loaders()
    model = SimplerNN()
    model.train(train_loader, val_loader, test_loader)
