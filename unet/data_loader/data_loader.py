import torch.nn as torch_nn
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Dataset.

    Example
    -------
    >>> from unet.data_loader.data_loader import ImageDataset
    >>> from torchvision import transforms
    >>> from torch.utils.data import DataLoader
    >>> from torchvision.transforms import ToTensor
    >>>
    >>> transform = transforms.Compose(
    ...    [
    ...        ToTensor(),
    ...    ]
    ... )
    >>> dataset_train = ImageDataset(
    ...    input_data=train_inputs_flatten,
    ...    output_data=train_outputs_flatten,
    ...    transform=transform,
    ... )
    >>> loader_train = DataLoader(
    ...    dataset_train,
    ...    batch_size=batch_size,
    ...    shuffle=True,
    ...    num_workers=4,
    ... )
    """

    def __init__(
        self,
        input_data,
        output_data,
        transform: torch_nn.Module = None,
    ) -> None:
        self.input_data = input_data
        self.output_data = output_data
        self.transform = transform

    def __len__(self) -> int:
        """Length."""
        return len(self.input_data)

    def __getitem__(self, index):
        """Get item."""
        input = self.input_data[index]
        output = self.output_data[index]

        if self.transform:
            input = self.transform(input)
            output = self.transform(output)

        return input, output


class TaskImageDataset(Dataset):
    """Dataset."""

    def __init__(
        self,
        input_data,
        output_data,
        transform: torch_nn.Module = None,
    ) -> None:
        self.input_data = input_data
        self.output_data = output_data
        self.transform = transform

    def __len__(self) -> int:
        """Length."""
        return len(self.input_data)

    def __getitem__(
        self,
        task_name: str,
        index: int,
    ):
        """Get item."""
        input = self.input_data[task_name][index]
        output = self.output_data[task_name][index]

        if self.transform:
            input = self.transform(input)
            output = self.transform(output)

        return input, output
