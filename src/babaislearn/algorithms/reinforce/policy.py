from torch import nn

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        action_count = 6
        self.layers = nn.Sequential(
            nn.Linear(594, 594),
            nn.ReLU(),
            nn.Linear(594, 594),
            nn.ReLU(),
            nn.Linear(594, 594),
            nn.ReLU(),
            nn.Linear(594, action_count),
            nn.Softmax(dim=0)
        )

        self.log_probs = []
        self.rewards = []
        self.actions = []
        self.entropies = []

    def forward(self, x):
        return self.layers(x)