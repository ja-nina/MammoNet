import torch.nn as nn
from MammoNet.models.BaseModel import BaseModel


class SimpleNNModel(nn.Module):
    def __init__(self, num_classes, input_size=224):
        super(SimpleNNModel, self).__init__()
        self.name = "SimpleNNModel"

        # Calculate the flattened input dimension
        flattened_dim = 3 * input_size * input_size
        self.fc1 = nn.Linear(flattened_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class SimpleNN(BaseModel):
    def __init__(self, num_classes=2, input_size=224, **kwargs):
        model = SimpleNNModel(num_classes, input_size)
        super().__init__(model=model, num_classes=num_classes, model_name="SimpleNNModel", **kwargs)
