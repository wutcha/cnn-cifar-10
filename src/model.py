from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), 
            nn.Flatten(), 
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64,10)    
        )
    def forward(self, x):
        logits = self.stack(x)
        return logits