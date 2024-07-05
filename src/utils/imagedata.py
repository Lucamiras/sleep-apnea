import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_mean_and_std(dataset):
    """
    Calculate the mean and standard deviation of the images in the given dataset.

    Args:
        train_dir (str): The directory containing the training images.
        dataset (torch.utils.data.Dataset): The dataset object containing the images.
        classes (list): The list of classes in the dataset.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the images.
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    mean = 0.0
    std = 0.0
    total_samples = 0
    for images, _ in tqdm(dataloader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height*width)
        mean += images.mean(2).sum(0)  # sum up the mean of each channel
        std += images.std(2).sum(0)  # sum up the std of each channel
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std
        