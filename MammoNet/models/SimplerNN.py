import torch.nn as nn
from MammoNet.dataset.data_handler import DataHandler 
from MammoNet.models.SimpleNN import SimpleNNModel, SimpleNN

class SimplerNNModel(SimpleNNModel):
    def __init__(self, num_classes, input_size=224):
        super(SimplerNNModel, self).__init__(num_classes, input_size=input_size)
        self.name = "SimplerNNModel"
        
        flattened_dim = 3 * input_size * input_size

        self.fc1 = nn.Linear(flattened_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimplerNN(SimpleNN):
    def __init__(self, num_classes=2, input_size=224):
        super(SimplerNN, self).__init__(num_classes=num_classes, input_size=input_size)
        self.model = SimplerNNModel(num_classes, input_size)
        
if __name__ == '__main__':
    data_handler = DataHandler()
    train_loader, val_loader, test_loader = data_handler.get_dataset_loaders()
    model = SimplerNN()
    model.train(train_loader, val_loader, test_loader)
