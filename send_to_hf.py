import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class MyModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dict):
        super().__init__()
        self.param = nn.Parameter(torch.rand(config["num_channels"], config["hidden_size"]))
        self.linear = nn.Linear(config["hidden_size"], config["num_classes"])

    def forward(self, x):
        return self.linear(x + self.param)

# create model
config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
model = MyModel(config=config)

# save locally
model.save_pretrained("my-awesome-model", config=config)

# push to the hub
model.push_to_hub("my-awesome-model", config=config)

# reload
model = MyModel.from_pretrained("username/my-awesome-model")