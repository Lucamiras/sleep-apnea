import os
import numpy as np
from torchvision import transforms
from PIL import Image
from typing_extensions import override
from torch import optim
import torch
from src.nets.train_eval_loop import train_epoch, eval_model
from src.nets.basicCNN import CNN
from src.dataset import SignalDataset
from torch.utils.data import DataLoader
from src.utils.imagedata import get_mean_and_std
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as f
import h5py
import json
from src.preprocess import (
    Config,
    Downloader,
    Extractor,
    Processor,
    Serializer,
    DataPreprocessor,
)
from src.utils.globals import (
    CLASSES
)

# Initialize pipeline elements

overrides = {
    "ids_to_process":['00000995'],
    "augment_ratio":0.5,
}

config = Config(
    classes=CLASSES,
    download_files=False,
    extract_signals=False,
    process_signals=True,
    serialize_signals=True,
    overrides=overrides,
)
downloader = Downloader(config)
extractor = Extractor(config)
processor = Processor(config)
serializer = Serializer(config)

# Initialize full pipeline
pre = DataPreprocessor(
    downloader,
    extractor,
    processor,
    serializer,
    config)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

ds = SignalDataset(os.path.join(config.signals_path, 'dataset.h5'), 'Train', transform=transform, classes=CLASSES)
dl = DataLoader(ds, batch_size=len(ds), shuffle=False)

means, stds = [],[]
for images, labels in dl:
    means.append(images.mean())
    stds.append(images.std())
    break

mean, std = means[0], stds[0]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean, std)
])

train = SignalDataset(os.path.join(config.signals_path, 'dataset.h5'), 'Train', transform=transform, classes=CLASSES)
valid = SignalDataset(os.path.join(config.signals_path, 'dataset.h5'), 'Val', transform=transform, classes=CLASSES)
test = SignalDataset(os.path.join(config.signals_path, 'dataset.h5'), 'Test', transform=transform, classes=CLASSES)

train_loader = DataLoader(train, batch_size=16, shuffle=True)
val_loader = DataLoader(valid, batch_size=16, shuffle=False)
test_loader = DataLoader(test, batch_size=16, shuffle=False)

class CNN_RNN(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_classes, slice_width=20, stride=10):
        super(CNN_RNN, self).__init__()

        # CNN layers
        self.cnn = CNN()

        self.rnn = nn.LSTM(cnn_output_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

        self.slice_width = slice_width
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        cnn_features = []
        for t in range(0, width, 28):
            x_slice = x[:, :, :, t:t+28]
            cnn_out = self.cnn(x_slice)
            cnn_out = cnn_out.view(batch_size, -1)
            cnn_features.append(cnn_out)

        cnn_features = torch.stack(cnn_features, dim=1)

        rnn_out, (hn, _) = self.rnn(cnn_features)

        out = self.fc(hn[-1])
        return out

model = CNN_RNN(12544, 1024, 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer = optim.Adam(model.parameters(), lr=4e-1)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
for epoch in range(10):
    train_loss, train_f1 = train_epoch(model, train_loader, optimizer=optimizer, criterion=criterion, device=device)
    print(f"{train_loss:.2f} -> {train_f1:.2f}")



