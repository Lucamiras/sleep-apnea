import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3, input_size=224):
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * int(self.input_size / 2 / 2 / 2) * int(self.input_size / 2 / 2 / 2), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = f.relu(x)
        x = self.pool(x)
        return x

class CNNtoRNN(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_classes, slice_width=16, stride=10):
        super(CNNtoRNN, self).__init__()
        self.cnn = CNN(num_classes)
        self.rnn = nn.LSTM(cnn_output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.slice_width = slice_width
        self.stride = stride

    def forward(self, x):
        #import pdb; pdb.set_trace()
        batch_size, channels, height, width = x.size()
        cnn_features = []
        for t in range(0, width, self.slice_width):
            x_slice = x[:, :, :, t:t+28]
            cnn_out = self.cnn(x_slice)
            cnn_out = cnn_out.view(batch_size, -1)
            cnn_features.append(cnn_out)
        cnn_features = torch.stack(cnn_features, dim=1)
        rnn_out, (hn, _) = self.rnn(cnn_features)
        out = self.fc(hn[-1])
        return out
