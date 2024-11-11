import torch
import torch.nn as nn
from MammoNet.models.BaseModel import BaseModel


class SimpleCNNModel(nn.Module):
    def __init__(self, num_classes, input_size=224):
        super(SimpleCNNModel, self).__init__()
        self.name = "SimpleCNNModel"
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        example_input = torch.zeros(1, 3, input_size, input_size)
        example_output = self._forward_conv(example_input)
        flattened_dim = example_output.view(-1).shape[0]

        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleCNN(BaseModel):
    def __init__(self, num_classes=2, input_size=224):
        model = SimpleCNNModel(num_classes, input_size)
        super().__init__(model, num_classes, "SimpleCNN", lr=0.001, epochs=10)
