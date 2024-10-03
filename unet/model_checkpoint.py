import os

import torch
import torch.nn as nn


def save(
    checkpoint_directory: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """
    Save the network and optimizer state.

    Parameters
    ----------
    checkpoint_directory : str
        Directory where the checkpoint will be saved.
    network : nn.Module
        The neural network model.
    optimizer : torch.optim.Optimizer
        The optimizer for the model.
    epoch : int
        The current epoch number.

    Examples
    --------
    >>> from unet.model_checkpoint import save
    >>> save(checkpoint_directory, network, optimizer, epoch)
    """
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    torch.save(
        {
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        os.path.join(checkpoint_directory, f"model_epoch{epoch}.pth"),
    )


def load(
    checkpoint_directory: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    """
    Load the network and optimizer state.

    Parameters
    ----------
    checkpoint_directory : str
        Directory where the checkpoint is saved.
    network : nn.Module
        The neural network model.
    optimizer : torch.optim.Optimizer
        The optimizer for the model.

    Returns
    -------
    tuple
        The network, optimizer, and the epoch number.

    Examples
    --------
    >>> from unet.model_checkpoint import load
    >>> network, optimizer, epoch = load(checkpoint_directory, network, optimizer)
    """
    if not os.path.exists(checkpoint_directory):
        epoch = 0
        return network, optimizer, epoch

    checkpoint_list = os.listdir(checkpoint_directory)
    checkpoint_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dictionary_model = torch.load(os.path.join(checkpoint_directory, checkpoint_list[-1]))

    network.load_state_dict(dictionary_model['network'])
    optimizer.load_state_dict(dictionary_model['optimizer'])
    epoch = int(checkpoint_list[-1].split('epoch')[1].split('.pth')[0])

    return network, optimizer, epoch
