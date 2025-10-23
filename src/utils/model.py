from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys
import numpy as np
import torch.nn as nn
import torch


def classification_report(y_true, y_pred):
    """
    Generate performance metrics for classification models.
    """
    try:
        logging.info("Calculating classification metrics...")

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        logging.info("Classification metrics calculated successfully.")

        return {
            " accuracy_score": acc,
            " f1_score": f1,
            " precision_score": precision,
            " recall_score": recall
        }

    except Exception as e:
        raise CustomException(e, sys)


def regression_report(y_true, y_pred):
    """
    Generate performance metrics for regression models.
    """
    try:
        logging.info("Calculating regression metrics...")

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        logging.info("Regression metrics calculated successfully.")

        return {
            " mean_squared_error": mse,
            " root_mean_squared_error": rmse,
            " r2_score": r2
        }

    except Exception as e:
        raise CustomException(e, sys)
    
class LSTMNet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers):
        super().__init__()

        # LSTM Layer
        self.gru = nn.LSTM(
            input_size=input_size,
            hidden_size=num_hidden,
            num_layers=num_layers,
            batch_first=True  # make input shape (batch, seq, features)
        )

        # Linear layer for output
        self.output = nn.Linear(num_hidden, 1)

        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Run through the GRU layer
        out, hidden = self.gru(x)  # out shape: (batch, seq_len, hidden_size)


        # Pass through linear layer
        out = self.output(out)

        # Apply sigmoid activation for BCELoss
        out = self.sigmoid(out)

        return out, hidden

    
def train_bridged_model(train_loader):
        input_size = 23
        num_hidden = 20
        num_layers = 10

        net = LSTMNet(input_size,num_hidden,num_layers)
        num_epochs = 20
        learning_rate = 0.001

        lossFun = torch.nn.BCELoss()
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)


        for _ in range(num_epochs):
            net.train()
            for X, y in train_loader:

                y_pred, _ = net(X)
                loss = lossFun(y_pred.squeeze(), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return net




def get_bridge_img_model_optimizer():
    net = ResNetLike(num_classes=1)
    lossFun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    return net, optimizer, lossFun

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetLike(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x = self.pool(x)                        # [B, 64, H/4, W/4]
        x = self.layer1(x)                      # -> [B, 64, H/4, W/4]
        x = self.layer2(x)                      # -> [B, 128, H/8, W/8]
        x = self.layer3(x)                      # -> [B, 256, H/16, W/16]
        x = self.layer4(x)                      # -> [B, 512, H/32, W/32]
        x = self.global_pool(x)                 # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)                 # -> [B, 512]
        x = self.fc(x)  # Logits

        return x
    
def create_target(row):
    # Map image prediction
    image_cond = 'Good' if row['bridge_image_prediction'] == 0 else 'Bad'
    
    # Map RUL
    if row['rul_model_prediction'] >= 180:
        rul_cond = 'Good'
    elif 50 < row['rul_model_prediction'] < 180:
        rul_cond = "Moderate"
    else:
        rul_cond = 'Bad'
        
    
    
    # Map bridged prediction
    struct_cond = 'Standing' if row['bridge_sensor_prediction'] == 0 else 'Collapsed'
    
    # Fusion: Prioritize negatives

    if 'Bad' in [image_cond, rul_cond] or struct_cond == 'Collapsed':
        return 0 # Bad
    elif image_cond == 'Good' and rul_cond == 'Good' and struct_cond == 'Standing':
        return 2 # Good
    elif image_cond == 'Good' and rul_cond == "Moderate" and struct_cond == 'Standing':
        return 1 # Moderate
