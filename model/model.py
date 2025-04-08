import torch.nn as nn
import torch # For building the networks

class NetFNNCNN_good(nn.Module):
    def __init__(self, out_features=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 32)

        self.fc2 = nn.Linear(32+4, out_features)

        self.fnn_fc1 = nn.Linear(2, 4)

    def forward(self, x_left, x_demo):
        x_left = self.conv1(x_left)
        x_left = self.conv2(x_left)
        x_left = self.conv3(x_left)
        x_left = x_left.view(x_left.shape[0], -1)
        x_left = self.relu(self.fc1(x_left))

        output_demo = self.relu(self.fnn_fc1(x_demo))

        output_concat = torch.cat((x_left, output_demo), axis=1)
        x_concat = self.fc2(output_concat)
        return x_concat
