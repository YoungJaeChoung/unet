import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class Trainer:
    """
    A class for training models.

    Parameters
    ----------
    model : nn.Module
        The model to be trained
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    save_path : str, optional
        Path to save the model (default: None)
    learning_rate : float, optional
        Learning rate (default: 0.001)

    Examples
    --------
    >>> from unet.train import Trainer
    >>> from unet.unet import UNet
    >>> model = UNet()
    >>> trainer = Trainer(model, train_loader, val_loader, save_path='model.pth')
    >>> trainer.train(num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str = None,
        learning_rate: float = 0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def train(self, num_epochs: int) -> None:
        """
        Train the model for the specified number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train

        Returns
        -------
        None
        """
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.scheduler.step(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if self.save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f'Model saved to {self.save_path}')

            print()  # Add a blank line between epochs

    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        """
        Validate the model for one epoch.

        Returns
        -------
        float
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)
