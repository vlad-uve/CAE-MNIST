
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_train_dataloader(batch_size=32, data_dir='../data'):
    """
    Loads the MNIST training set and returns a DataLoader.

    Args:
        batch_size (int): size of each training batch
        data_dir (str): path to MNIST data storage

    Returns:
        DataLoader: PyTorch DataLoader for training data
    """
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataloader


def get_validation_dataloader(batch_size=500, data_dir='../data'):
    """
    Loads the MNIST validation (test) set and returns a DataLoader.

    Args:
        batch_size (int): size of each validation batch
        data_dir (str): path to MNIST data storage

    Returns:
        DataLoader: PyTorch DataLoader for validation data
    """
    validation_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transforms.ToTensor()
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return validation_dataloader
