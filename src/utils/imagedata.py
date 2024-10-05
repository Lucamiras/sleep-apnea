from torch.utils.data import DataLoader
from src.dataset import SignalDataset
from torchvision import transforms


def get_mean_and_std(path:str, input_size:tuple, classes:dict):
    """
    Calculate the mean and standard deviation of the images in the given dataset.

    Args:
        path (str): The directory containing the training images.
        input_size (tuple): Size of the images
        classes (dict): The list of classes in the dataset.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the images.
    """
    dataset_pre_norm = SignalDataset(
        path,
        transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ]),
        classes=classes
    )

    dataloader = DataLoader(
        dataset_pre_norm,
        batch_size=len(dataset_pre_norm),
        shuffle=False
    )

    for images, _ in dataloader:
        mean = images.mean()
        std = images.std()

    return mean, std
        