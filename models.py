import torch
from torch import nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, n_classes, Chans=128, Samples=768, dropoutRate=0.5, kernLength=64, F1=8, D=1):
        super(EEGNet, self).__init__()
        F2 = D * F1

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        dummy_input = torch.zeros(1, 1, Chans, Samples)
        x = self.firstconv(dummy_input)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        self.flattened_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, n_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class MidCNN(nn.Module):
    def __init__(self, n_output, input_shape, dropout_rate, fc1_units=128, fc_drop=None, **kwargs):
        super().__init__()
        chans, samples = input_shape

        self.channel_weights = nn.Parameter(torch.ones(1, chans, 1))
        self.spatial = nn.Conv2d(1, 4, (chans, 1), bias=False)

        self.conv1 = nn.Conv1d(4, 64, kernel_size=50, padding=25)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=25, padding=12)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(fc_drop if fc_drop is not None else dropout_rate)
        self.fc1 = nn.Linear(256, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_output)

    def forward(self, x):
        x = x.squeeze(1)
        x = x * self.channel_weights
        x = x.unsqueeze(1)
        x = self.spatial(x).squeeze(2)
        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        x = self.pool3(F.gelu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)



class DeeperTemporalCNN(nn.Module):
    def __init__(self, n_output, input_shape, dropout_rate, fc1_units=128, fc2_units=0, fc_drop=None, **kwargs):
        super().__init__()
        chans, samples = input_shape

        self.channel_weights = nn.Parameter(torch.ones(1, chans, 1))  # (1, C, 1)

        self.conv1 = nn.Conv1d(chans, 32, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.gpool = nn.AdaptiveAvgPool1d(1)

        final_dropout = fc_drop if fc_drop is not None else dropout_rate
        self.dropout = nn.Dropout(final_dropout)

        self.fc1 = nn.Linear(256, fc1_units)
        # self.bn_fc1 = nn.BatchNorm1d(fc1_units)

        if fc2_units > 0:
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc_out = nn.Linear(fc2_units, n_output)
        else:
            self.fc2 = None
            self.fc_out = nn.Linear(fc1_units, n_output)

    def forward(self, x):
        x = x.squeeze(1)                 # (B, 1, C, T) → (B, C, T)
        x = x * self.channel_weights     # Channel-wise scaling

        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        x = self.pool3(F.gelu(self.bn3(self.conv3(x))))
        x = self.gpool(F.gelu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))

        if self.fc2:
            x = F.relu(self.fc2(x))

        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)


def init_sherpa_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SherpaModel(nn.Module):
    def __init__(self, n_output, input_shape, dropout_rate=0.2, fc_drop=0.5,
                 fc1_units=128, fc2_units=64, fc3_units=32,
                 kernel_size=50, conv_filters=[16, 32, 64, 128]):
        super().__init__()
        chans, timepoints = input_shape
        padding = kernel_size // 2

        self.batchnorm_input = nn.BatchNorm1d(chans)

        self.conv1 = nn.Conv1d(chans, conv_filters[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv1d(conv_filters[2], conv_filters[3], kernel_size=kernel_size, padding=padding)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batchnorm_mid = nn.BatchNorm1d(conv_filters[3])

        self.gelu = nn.GELU(approximate='tanh')
        self.relu = nn.ReLU()

        # Note: Only 3x maxpool → downsample by 8x
        # flattened_size = conv_filters[3] * (timepoints // (2 ** 3))
        with torch.no_grad():
            dummy_input = torch.zeros(1, chans, timepoints)
            x = self.maxpool(self.gelu(self.conv1(dummy_input)))
            x = self.maxpool(self.gelu(self.conv2(x)))
            x = self.maxpool(self.gelu(self.conv3(x)))
            x = self.gelu(self.conv4(x))
            x = self.batchnorm_mid(x)
            flattened_size = x.view(1, -1).size(1)

        # Dropout before FC layers
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_fc1 = nn.Dropout(fc_drop)
        self.dropout_fc2 = nn.Dropout(fc_drop)

        # Fully connected stack
        self.fc1 = nn.Linear(flattened_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.out = nn.Linear(fc3_units, n_output)


    def forward(self, x):
        # x = x.squeeze(1)  # (B, 1, C, T) → (B, C, T)

        x = self.batchnorm_input(x)
        x = self.maxpool(self.gelu(self.conv1(x)))
        x = self.maxpool(self.gelu(self.conv2(x)))
        x = self.maxpool(self.gelu(self.conv3(x)))
        x = self.gelu(self.conv4(x))  # No pooling here

        x = self.batchnorm_mid(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.dropout_fc1(self.relu(self.fc1(x)))
        x = self.dropout_fc2(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x  # raw logits



class SoftmaxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return F.softmax(self.model(x), dim=1)