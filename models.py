import torch, torch.nn as nn

class IndexLSTM(nn.Module):            # 29,657 trainable parameters
    def __init__(self, hidden=84):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 5)
    def forward(self, x):              # x: (B, 24, 1)
        o, (h, c) = self.lstm(x)
        return self.head(h[-1])

class MapCNN(nn.Module):               # 30,229 trainable parameters
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((3, 6)),
            nn.Flatten(), nn.Linear(288, 32), nn.ReLU(), nn.Linear(32, 5))
    def forward(self, x):              # x: (B, 24, 21, 81)
        return self.net(x)

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
