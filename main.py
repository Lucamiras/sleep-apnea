from src.dataset import MelSpectrogramDataset
from src.preprocess import Preprocessor
from src.preprocess import get_download_urls
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

run_pre = False

if run_pre:
    pre = Preprocessor(
        project_dir='data',
        edf_urls=EDF_URLS,
        rml_urls=RML_URLS,
        data_channels=DATA_CHANNELS,
        classes=CLASSES,
        clip_length=30,
        sample_rate=48000,
        ids_to_process=['00000995']
    )

    pre.run(
        download=False,
        dictionary=True,
        segments=True,
        create_files=False,
        mel_frequencies_dict=True,
        shuffle=False
    )

INPUT_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    ])

train_dataset = MelSpectrogramDataset(
    os.path.join('data', 'processed', 'signals'),
    transform=transform,
    classes=CLASSES)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch, labels in train_loader:
    image = batch[0]
    sample_image_np = image.permute(1, 2, 0).numpy()  # Change shape from [C, H, W] to [H, W, C]

    # Plot the image
    plt.imshow(sample_image_np)
    plt.axis('off')  # Hide the axes
    plt.show()
    break

