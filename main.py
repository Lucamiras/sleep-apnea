from src.preprocess import Preprocessor
from src.dataset import SpectrogramDataset
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os
from torchvision import transforms
from src.utils.imagedata import get_mean_and_std
import librosa

warnings.filterwarnings('ignore')

spectrogram_train_path = os.path.join('data', 'processed', 'spectrogram','train')
spectrogram_val_path = os.path.join('data', 'processed', 'spectrogram','validation')
spectrogram_test_path = os.path.join('data', 'processed', 'spectrogram','test')

num_train_examples = len(os.listdir(spectrogram_train_path))
num_val_examples = len(os.listdir(spectrogram_val_path))
num_test_examples = len(os.listdir(spectrogram_test_path))

img_size = (256, 256)

transform_without_normalization = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
    ])

train_data_not_norm = SpectrogramDataset(spectrogram_train_path, transform=transform_without_normalization, classes=CLASSES)
mean, std = get_mean_and_std(train_data_not_norm)
mean = list(mean.numpy())
std = list(std.numpy())
print(mean, std)
