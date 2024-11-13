import torch.nn as nn
from MammoNet.models.BaseModel import BaseModel


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


class SimplerNN(BaseModel):
    def __init__(self, num_classes=2, input_size=224, lr=0.001, epochs=10):
        model = SimplerNNModel(num_classes, input_size)
        super().__init__(model=model, num_classes=num_classes, model_name="SimplerNNModel", lr=lr, epochs=epochs)
