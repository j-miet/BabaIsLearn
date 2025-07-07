from torch import nn

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(594, 594),
            nn.ReLU(),
            nn.Linear(594, 594),
            nn.ReLU(),
            nn.Linear(594, 594),
            nn.ReLU(),
            nn.Linear(594, 1)
        )

        self.values = []

    def forward(self, x):
        return self.layers(x)